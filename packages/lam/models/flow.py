"""
Optical Flow Supervision for LAM.

Provides motion-enriched latent representations by training the VAE's latent
to predict optical flow between frames. Uses online knowledge distillation
from a frozen RAFT teacher model.

Components:
- RAFTTeacher: Frozen optical flow model for ground-truth generation
- FlowDecoder: Transformer that predicts flow from latent + context image
"""

import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)

# Supported RAFT variants
FlowModelType = Literal["raft_small", "raft_large"]


@dataclass
class FlowConfig:
    """Configuration for flow supervision.

    All fields are required when flow supervision is enabled.
    """

    model: FlowModelType
    loss_weight: float
    decoder_depth: int
    warmup_steps: int = 0  # Steps to linearly ramp up flow loss (0 = no warmup)
    # RAFT teacher performance knobs
    teacher_num_flow_updates: int = 12  # RAFT refinement iterations (smaller = faster)
    teacher_chunk_size: int = (
        64  # Teacher batch chunking (larger = faster, more memory)
    )
    # Optional global flow-summary auxiliary loss (off by default)
    summary_loss_weight: float = 0.0
    summary_static_eps: float = 1e-6

    def __post_init__(self):
        if self.model not in ("raft_small", "raft_large"):
            raise ValueError(
                f"flow.model must be 'raft_small' or 'raft_large', got '{self.model}'"
            )
        if self.loss_weight <= 0:
            raise ValueError(
                f"flow.loss_weight must be positive, got {self.loss_weight}"
            )
        if self.decoder_depth <= 0:
            raise ValueError(
                f"flow.decoder_depth must be positive, got {self.decoder_depth}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"flow.warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.teacher_num_flow_updates <= 0:
            raise ValueError(
                f"flow.teacher_num_flow_updates must be positive, got {self.teacher_num_flow_updates}"
            )
        if self.teacher_chunk_size <= 0:
            raise ValueError(
                f"flow.teacher_chunk_size must be positive, got {self.teacher_chunk_size}"
            )
        if self.summary_loss_weight < 0:
            raise ValueError(
                f"flow.summary_loss_weight must be non-negative, got {self.summary_loss_weight}"
            )
        if self.summary_static_eps <= 0:
            raise ValueError(
                f"flow.summary_static_eps must be positive, got {self.summary_static_eps}"
            )

    def get_weight(self, step: int) -> float:
        """Get effective loss weight at given training step.

        Linearly ramps from 0 to loss_weight over warmup_steps.
        """
        if self.warmup_steps == 0:
            return self.loss_weight

        warmup_factor = min(1.0, step / self.warmup_steps)
        return self.loss_weight * warmup_factor


class RAFTTeacher(nn.Module):
    """
    Frozen RAFT optical flow model for ground-truth generation.

    Lazily loads torchvision's RAFT model on first use.
    The RAFT model is intentionally not registered as a submodule to avoid
    checkpoint/state_dict pollution and unnecessary traversal overhead.

    Input normalization matches torchvision's RAFT weights preset:
    map image pixels from [0, 1] to [-1, 1].
    """

    def __init__(
        self,
        model_name: FlowModelType,
        chunk_size: int = 64,
        num_flow_updates: int = 12,
    ):
        super().__init__()
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._num_flow_updates = num_flow_updates

        # Intentionally kept out of nn.Module registration; see docstring.
        self._model_device: Optional[torch.device] = None
        self.__dict__["_model"] = None

    def _ensure_loaded(self, device: torch.device) -> nn.Module:
        model = self.__dict__.get("_model", None)
        if model is None:
            from torchvision.models.optical_flow import (
                raft_small,
                raft_large,
                Raft_Small_Weights,
                Raft_Large_Weights,
            )

            if self._model_name == "raft_small":
                weights = Raft_Small_Weights.DEFAULT
                model = raft_small(weights=weights)
                logger.info("Loaded RAFT-Small optical flow teacher")
            else:
                weights = Raft_Large_Weights.DEFAULT
                model = raft_large(weights=weights)
                logger.info("Loaded RAFT-Large optical flow teacher")

            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            # Avoid registering as a submodule.
            self.__dict__["_model"] = model

        if self._model_device != device:
            model = model.to(device)
            self.__dict__["_model"] = model
            self._model_device = device

        return model

    @torch.inference_mode()
    def compute_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        *,
        num_flow_updates: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame [B, C, 1, H, W] in [0, 1] range
            frame2: Second frame [B, C, 1, H, W] in [0, 1] range
            num_flow_updates: Optional RAFT refinement iterations override

        Returns:
            flow: Optical flow field [B, 2, H, W] (dx, dy per pixel)
        """
        model = self._ensure_loaded(frame1.device)

        # Remove time dimension: [B, C, 1, H, W] -> [B, C, H, W]
        img1 = frame1.squeeze(2)
        img2 = frame2.squeeze(2)

        # Run RAFT teacher in full precision to avoid AMP-related cuDNN/grid_sample issues
        # (this also overrides any outer autocast context, e.g. from Lightning AMP).
        device_type = "cuda" if frame1.is_cuda else "cpu"
        autocast_off = (
            torch.autocast(device_type=device_type, enabled=False)
            if hasattr(torch, "autocast")
            else nullcontext()
        )
        chunk_size = self._chunk_size if self._chunk_size > 0 else img1.shape[0]
        flow_chunks: list[torch.Tensor] = []

        with autocast_off:
            # Process in smaller chunks to keep RAFT internals/correlation volumes within
            # cuDNN-supported limits.
            for start in range(0, img1.shape[0], chunk_size):
                end = start + chunk_size
                img1_chunk = img1[start:end]
                img2_chunk = img2[start:end]

                # Normalize like torchvision.transforms._presets.OpticalFlow:
                # - Ensure float32
                # - map [0, 1] -> [-1, 1]
                img1_t = (
                    img1_chunk.to(dtype=torch.float32).mul(2.0).sub(1.0).contiguous()
                )
                img2_t = (
                    img2_chunk.to(dtype=torch.float32).mul(2.0).sub(1.0).contiguous()
                )

                # RAFT returns list of flow predictions at different refinement levels
                # Take the last (most refined) prediction
                updates = (
                    self._num_flow_updates
                    if num_flow_updates is None
                    else num_flow_updates
                )
                flow_predictions = model(img1_t, img2_t, num_flow_updates=updates)
                flow_chunks.append(flow_predictions[-1].float())

        return torch.cat(flow_chunks, dim=0)

    def state_dict(self, *args, **kwargs):
        """Override to prevent RAFT weights from being saved."""
        return {}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Override to skip loading (RAFT is always loaded fresh)."""
        pass


class FlowDecoder(nn.Module):
    """
    Transformer decoder that predicts optical flow from latent action + context image.

    Architecture mirrors the auxiliary pixel decoder:
    - Input: Pixel context tokens from first frame (spatial layout)
    - Cross-attention context: Quantized latent action (motion encoding)
    - Output: Dense optical flow field [B, 2, H, W]

    The latent encodes "what motion happened" while the image provides
    "where objects are" - together they predict "which pixels moved where".
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        image_size: tuple[int, int],
        effective_grid_size: tuple[int, int],
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        """
        Args:
            dim: Transformer embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per attention head
            image_size: Output image size (H, W)
            effective_grid_size: Spatial grid size after encoding (h, w)
            attn_dropout: Attention dropout rate
            ff_dropout: Feed-forward dropout rate
        """
        super().__init__()

        from lam.models.attention import Transformer

        self.dim = dim
        self.effective_grid_size = effective_grid_size

        # Transformer with cross-attention for action conditioning
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
            has_cross_attn=True,
            dim_context=dim,
        )

        # Output projection: dim -> 2 channels (dx, dy) per patch
        image_height, image_width = image_size
        eff_h, eff_w = effective_grid_size
        patch_h = image_height // eff_h
        patch_w = image_width // eff_w

        self.to_flow = nn.Sequential(
            nn.Linear(dim, 2 * patch_h * patch_w),
            Rearrange(
                "b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)", p1=patch_h, p2=patch_w, c=2
            ),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict optical flow from context + action.

        Args:
            context_tokens: Pixel context from first frame [B, 1, h, w, d]
            action_tokens: Quantized latent action [B, 1, h', w', d]
            attn_bias: Spatial attention bias

        Returns:
            pred_flow: Predicted optical flow [B, 2, H, W]
        """
        b = context_tokens.shape[0]
        h, w = self.effective_grid_size

        video_shape = tuple(context_tokens.shape[:-1])

        # Flatten for transformer
        context_flat = rearrange(context_tokens, "b t h w d -> (b t) (h w) d")
        action_flat = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")

        # Run transformer with cross-attention
        out = self.transformer(
            context_flat,
            attn_bias=attn_bias,
            video_shape=video_shape,
            context=action_flat,
        )

        # Reshape and project to flow
        out = rearrange(out, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        pred_flow = self.to_flow(out).squeeze(2)  # [B, 2, H, W]

        return pred_flow


def compute_flow_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute MSE loss between predicted and ground-truth flow.

    Flow is normalized by image dimensions to put values in ~[-1, 1] range,
    making the loss scale comparable to other losses (DINO, pixel).

    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground-truth flow from RAFT [B, 2, H, W]
        normalize: If True, normalize flow by image size before loss

    Returns:
        loss: Scalar MSE loss
    """
    pred_flow, gt_flow = _normalize_flow_for_loss(
        pred_flow=pred_flow,
        gt_flow=gt_flow,
        normalize=normalize,
    )

    return F.mse_loss(pred_flow, gt_flow)


def compute_weighted_mean_flow(
    flow_b2hw: torch.Tensor,
    static_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample magnitude-weighted average flow direction.

    Args:
        flow_b2hw: Optical flow [B, 2, H, W]
        static_eps: Threshold for static fallback and denominator stability

    Returns:
        (mean_dx, mean_dy): Each shape [B]
    """
    if static_eps <= 0:
        raise ValueError(f"static_eps must be positive, got {static_eps}")

    fx = flow_b2hw[:, 0]
    fy = flow_b2hw[:, 1]
    mag = torch.sqrt(fx * fx + fy * fy)
    total_mag = mag.flatten(1).sum(dim=1)
    weights = mag / (total_mag[:, None, None] + static_eps)

    mean_dx = (fx * weights).flatten(1).sum(dim=1)
    mean_dy = (fy * weights).flatten(1).sum(dim=1)

    is_static = total_mag <= static_eps
    if is_static.any():
        fallback_dx = fx.flatten(1).mean(dim=1)
        fallback_dy = fy.flatten(1).mean(dim=1)
        mean_dx = torch.where(is_static, fallback_dx, mean_dx)
        mean_dy = torch.where(is_static, fallback_dy, mean_dy)

    return mean_dx, mean_dy


def compute_flow_summary_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    static_eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute MSE between magnitude-weighted global flow vectors.

    Args:
        pred_flow: Predicted flow [B, 2, H, W]
        gt_flow: Ground-truth flow [B, 2, H, W]
        static_eps: Threshold for static fallback and denominator stability
        normalize: If True, normalize flow by image size before summary extraction

    Returns:
        loss: Scalar MSE over [mean_dx, mean_dy]
    """
    pred_flow, gt_flow = _normalize_flow_for_loss(
        pred_flow=pred_flow,
        gt_flow=gt_flow,
        normalize=normalize,
    )

    pred_mean_dx, pred_mean_dy = compute_weighted_mean_flow(
        pred_flow, static_eps=static_eps
    )
    gt_mean_dx, gt_mean_dy = compute_weighted_mean_flow(gt_flow, static_eps=static_eps)

    pred_mean = torch.stack((pred_mean_dx, pred_mean_dy), dim=1)
    gt_mean = torch.stack((gt_mean_dx, gt_mean_dy), dim=1)
    return F.mse_loss(pred_mean, gt_mean)


def _normalize_flow_for_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not normalize:
        return pred_flow, gt_flow

    # Flow channel 0 = dx (horizontal), normalize by W
    # Flow channel 1 = dy (vertical), normalize by H
    _, _, h, w = gt_flow.shape
    norm = gt_flow.new_tensor([w, h]).view(1, 2, 1, 1)
    return pred_flow / norm, gt_flow / norm
