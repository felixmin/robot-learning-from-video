"""
Flow visualization strategy for LAQ validation.

Visualizes predicted vs ground-truth optical flow to monitor flow decoder learning.
"""

from typing import Any, Dict

import math
import torch
from torchvision.utils import make_grid, flow_to_image
from torchvision.transforms.functional import pil_to_tensor
from einops import rearrange
import lightning.pytorch as pl
from PIL import Image, ImageDraw

from .core import ValidationStrategy, ValidationCache


class FlowVisualizationStrategy(ValidationStrategy):
    """
    Visualize optical flow predictions vs RAFT ground truth.

    Creates side-by-side comparisons:
    - Frame t (source)
    - Frame t+k (target)
    - Ground-truth flow (from RAFT)
    - Predicted flow (from flow decoder)
    - Mean-flow direction panel (GT vs predicted arrows)

    Only runs if the model has flow supervision enabled.
    """

    def __init__(
        self,
        name: str = "flow_visualization",
        enabled: bool = True,
        num_samples: int = 8,
        every_n_validations: int = 1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            **kwargs,
        )
        self.num_samples = num_samples

    def needs_caching(self) -> bool:
        return True  # Need frames for visualization

    def needs_codes(self) -> bool:
        return False

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate flow visualization comparing predicted vs ground-truth."""
        wandb_logger = self._get_wandb_logger(trainer)

        if wandb_logger is None:
            return self.no_output("wandb_logger_unavailable")

        # Check if model has flow decoder
        model = pl_module.model
        if model.flow_decoder is None or model.flow_teacher is None:
            return self.no_output("flow_decoder_or_teacher_unavailable")

        # Get frames from cache
        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) == 0:
            return self.no_output("no_cached_frames")

        # Sample frames
        n_samples = min(self.num_samples, len(all_frames))
        indices = torch.randperm(len(all_frames))[:n_samples]
        frames = all_frames[indices]

        # Generate flow visualization
        flow_grid = self._create_flow_grid(frames, pl_module)
        if flow_grid is None:
            return self.no_output("flow_grid_creation_failed")

        bucket_name = cache.bucket_name or ""
        prefix = f"val/{bucket_name}" if bucket_name else "val"
        wandb_logger.log_image(
            key=f"{prefix}/flow_comparison{metric_suffix}",
            images=[flow_grid],
            caption=[f"Step {trainer.global_step} (GT flow | Pred flow | mean-flow direction)"],
        )
        return self.success(produced=1)

    def _create_flow_grid(
        self,
        frames: torch.Tensor,
        pl_module: pl.LightningModule,
    ) -> torch.Tensor:
        """Create visualization grid comparing GT and predicted flow."""
        if len(frames) == 0:
            return None

        model = pl_module.model
        device = pl_module.device

        was_training = model.training
        model.eval()
        with torch.no_grad():
            frames = frames.to(device)

            # Extract frame pairs
            first_frame = frames[:, :, :1]  # [B, C, 1, H, W]
            rest_frames = frames[:, :, 1:]  # [B, C, 1, H, W]

            # Get ground-truth flow from RAFT teacher
            gt_flow = model.flow_teacher.compute_flow(first_frame, rest_frames)

            # Get predicted flow from model
            # We need to run encoding and get the latent action
            enc_first_tokens, enc_rest_tokens, first_tokens, last_tokens = (
                model._encode_frames(first_frame, rest_frames)
            )
            tokens, _, _, _ = model.vq(first_tokens, last_tokens, codebook_training_only=False)

            action_h, action_w = model.action_shape
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=action_h, w=action_w)

            # Get pixel context
            dec_first_frame_tokens = model.decoder_context_projection(first_frame)

            # Compute attention bias
            h, w = model.patch_height_width
            attn_bias = model.spatial_rel_pos_bias(h, w, device=device)

            # Predict flow
            pred_flow = model.flow_decoder(
                dec_first_frame_tokens,
                tokens.detach(),
                attn_bias,
            )

        model.train(was_training)

        # Convert flow to RGB images using color wheel
        # flow_to_image expects [B, 2, H, W] and returns [B, 3, H, W] in [0, 255]
        gt_flow_rgb = flow_to_image(gt_flow).float() / 255.0
        pred_flow_rgb = flow_to_image(pred_flow).float() / 255.0

        from laq.models.flow import compute_weighted_mean_flow

        static_eps = float(getattr(model.flow_config, "summary_static_eps", 1e-6))
        gt_mean_dx, gt_mean_dy = compute_weighted_mean_flow(gt_flow, static_eps=static_eps)
        pred_mean_dx, pred_mean_dy = compute_weighted_mean_flow(pred_flow, static_eps=static_eps)

        # Get source and target frames
        frame_t = frames[:, :, 0].cpu()  # [B, C, H, W]
        frame_t_plus = frames[:, :, 1].cpu()

        gt_flow_rgb = gt_flow_rgb.cpu()
        pred_flow_rgb = pred_flow_rgb.cpu()
        gt_mean_dx = gt_mean_dx.cpu()
        gt_mean_dy = gt_mean_dy.cpu()
        pred_mean_dx = pred_mean_dx.cpu()
        pred_mean_dy = pred_mean_dy.cpu()

        direction_panels = []
        for i in range(frames.shape[0]):
            direction_panels.append(
                self._create_direction_panel(
                    gt_dx=float(gt_mean_dx[i].item()),
                    gt_dy=float(gt_mean_dy[i].item()),
                    pred_dx=float(pred_mean_dx[i].item()),
                    pred_dy=float(pred_mean_dy[i].item()),
                    height=frame_t.shape[-2],
                    width=frame_t.shape[-1],
                )
            )
        direction_panels_t = torch.stack(direction_panels, dim=0)

        # Stack: [frame_t, frame_t+k, gt_flow, pred_flow, direction_summary]
        imgs = torch.stack([frame_t, frame_t_plus, gt_flow_rgb, pred_flow_rgb, direction_panels_t], dim=0)
        imgs = rearrange(imgs, 'r b c h w -> (b r) c h w')
        imgs = imgs.clamp(0.0, 1.0)

        return make_grid(imgs, nrow=5, normalize=False)

    def _create_direction_panel(
        self,
        gt_dx: float,
        gt_dy: float,
        pred_dx: float,
        pred_dy: float,
        height: int,
        width: int,
    ) -> torch.Tensor:
        panel = Image.new("RGB", (int(width), int(height)), (18, 18, 18))
        draw = ImageDraw.Draw(panel)

        cx = width / 2.0
        cy = height / 2.0
        radius = 0.35 * min(width, height)

        gt_mag = math.hypot(gt_dx, gt_dy)
        pred_mag = math.hypot(pred_dx, pred_dy)
        max_mag = max(gt_mag, pred_mag)
        scale = 0.0 if max_mag <= 1e-12 else (radius / max_mag)

        draw.line([(0, cy), (width, cy)], fill=(60, 60, 60), width=1)
        draw.line([(cx, 0), (cx, height)], fill=(60, 60, 60), width=1)

        gt_end = (cx + gt_dx * scale, cy + gt_dy * scale)
        pred_end = (cx + pred_dx * scale, cy + pred_dy * scale)
        self._draw_arrow(draw, start=(cx, cy), end=gt_end, color=(80, 220, 80))
        self._draw_arrow(draw, start=(cx, cy), end=pred_end, color=(230, 90, 90))

        legend_top = max(2, int(height * 0.04))
        box = max(3, int(min(height, width) * 0.035))
        draw.rectangle([(4, legend_top), (4 + box, legend_top + box)], fill=(80, 220, 80))
        draw.rectangle([(4, legend_top + box + 3), (4 + box, legend_top + 2 * box + 3)], fill=(230, 90, 90))

        return pil_to_tensor(panel).float() / 255.0

    def _draw_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        color: tuple[int, int, int],
    ) -> None:
        draw.line([start, end], fill=color, width=3)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        mag = math.hypot(dx, dy)
        if mag <= 1e-12:
            return

        ux = dx / mag
        uy = dy / mag
        head_len = max(6.0, min(14.0, 0.22 * mag))
        spread = math.pi / 7.0

        left = (
            end[0] - head_len * (ux * math.cos(spread) + uy * math.sin(spread)),
            end[1] - head_len * (uy * math.cos(spread) - ux * math.sin(spread)),
        )
        right = (
            end[0] - head_len * (ux * math.cos(spread) - uy * math.sin(spread)),
            end[1] - head_len * (uy * math.cos(spread) + ux * math.sin(spread)),
        )
        draw.polygon([end, left, right], fill=color)
