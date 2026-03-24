"""
PyTorch Lightning task wrapper for LAM training.

Wraps LatentActionModel in a LightningModule with:
- LAPA-style optimizer (separate weight decay groups)
- Loss logging and codebook usage tracking
- Hydra configuration integration
- Optional EMA
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig

from common.batch_utils import (
    move_dataclass_tensors_to_device,
    select_primary_image_stream,
    temporal_frames_to_bcthw,
    uint8_image_streams_to_float32,
)
from lam.models.latent_action_model import LatentActionModel, DinoConfig
from lam.models.flow import FlowConfig
from common.lerobot_v3_types import Stage1Batch


def separate_weight_decayable_params(
    params: List[nn.Parameter],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Separate parameters into two groups for weight decay.

    Following LAPA convention:
    - 2D+ parameters (weights): apply weight decay
    - <2D parameters (biases, layernorms): no weight decay

    Args:
        params: All model parameters

    Returns:
        (wd_params, no_wd_params) tuple
    """
    wd_params = []
    no_wd_params = []

    for param in params:
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            wd_params.append(param)
        else:
            no_wd_params.append(param)

    return wd_params, no_wd_params


class LAMTask(pl.LightningModule):
    """
    PyTorch Lightning task for LAM training.

    Wraps LatentActionModel with training logic matching LAPA:
    - AdamW optimizer with separated weight decay groups
    - Cosine annealing LR scheduler with warmup
    - Reconstruction loss + codebook usage tracking
    - Visualization support via callbacks

    Args:
        model_config: Model configuration (DictConfig or dict)
        training_config: Training configuration (DictConfig or dict)
    """

    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
    ):
        super().__init__()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Store configs
        self.model_config = model_config
        self.training_config = training_config

        flow_cfg = model_config.flow
        flow_enabled = bool(flow_cfg.enabled)
        flow_config = None
        if flow_enabled:
            flow_config = FlowConfig(
                model=flow_cfg.model,
                loss_weight=flow_cfg.loss_weight,
                decoder_depth=flow_cfg.decoder_depth,
                warmup_steps=flow_cfg.warmup_steps,
                teacher_num_flow_updates=flow_cfg.teacher_num_flow_updates,
                teacher_chunk_size=flow_cfg.teacher_chunk_size,
                summary_loss_weight=flow_cfg.summary_loss_weight,
                summary_static_eps=flow_cfg.summary_static_eps,
            )

        dino_cfg = model_config.dino
        dino_enabled = bool(dino_cfg.enabled)
        dino_config = None
        if dino_enabled:
            dino_config = DinoConfig(
                loss_weight=float(dino_cfg.loss_weight),
                warmup_steps=int(dino_cfg.warmup_steps),
            )

        codebook_replace_schedule = [
            tuple(entry) for entry in model_config.codebook_replace_schedule
        ]
        vq_discarding_threshold_schedule = None
        if (
            "vq_discarding_threshold_schedule" in model_config
            and model_config.vq_discarding_threshold_schedule is not None
        ):
            vq_discarding_threshold_schedule = [
                tuple(entry) for entry in model_config.vq_discarding_threshold_schedule
            ]

        metrics_cfg = training_config.metrics

        # Initialize LAM model
        self.model = LatentActionModel(
            dim=model_config.dim,
            quant_dim=model_config.quant_dim,
            codebook_size=model_config.codebook_size,
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            spatial_depth=model_config.spatial_depth,
            temporal_depth=model_config.temporal_depth,
            dim_head=model_config.dim_head,
            heads=model_config.heads,
            code_seq_len=model_config.code_seq_len,
            vq_discarding_threshold=model_config.vq_discarding_threshold,
            vq_discarding_threshold_schedule=vq_discarding_threshold_schedule,
            channels=model_config.channels,
            attn_dropout=model_config.attn_dropout,
            ff_dropout=model_config.ff_dropout,
            latent_ablation=model_config.latent_ablation,
            use_dinov3_encoder=model_config.use_dinov3_encoder,
            dinov3_model_name=model_config.dinov3_model_name,
            dinov3_pool_to_grid=model_config.dinov3_pool_to_grid,
            metrics_num_unique_codes_every_n_steps=int(
                metrics_cfg.num_unique_codes_every_n_steps
            ),
            dino_config=dino_config,
            use_dino_decoder=dino_enabled,
            # Training decoder flags
            use_pixel_decoder=model_config.use_pixel_decoder,
            # Interpretability decoder flag
            use_aux_decoder=model_config.use_aux_decoder,
            flow_config=flow_config,
            codebook_replace_schedule=codebook_replace_schedule,
        )

        # Flag for one-time batch validation (to catch interface issues early)
        self._batch_validated = False

    @staticmethod
    def _extract_frames_from_stage1_batch(batch: Stage1Batch) -> torch.Tensor:
        return temporal_frames_to_bcthw(
            select_primary_image_stream(batch.image_streams),
            expected_time_steps=2,
        )

    @staticmethod
    def _batch_size_from_stage1_batch(batch: Stage1Batch) -> int:
        return int(select_primary_image_stream(batch.image_streams).shape[0])

    @staticmethod
    def _metric_value_for_log(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.ndim == 0 and not torch.is_floating_point(value):
                return value.to(torch.float32)
            return value
        if isinstance(value, (bool, int)):
            return float(value)
        return value

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ) -> Any:
        """
        Forward pass through LAM model.

        Returns:
            If return_recons_only: reconstructed frames [B, C, H, W]
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: (loss, metrics_dict)
        """
        return self.model(
            video,
            step=step,
            return_recons_only=return_recons_only,
            return_only_codebook_ids=return_only_codebook_ids,
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        Validate batch keys on first batch to catch interface issues early.

        This hook runs once per training to verify that the dataloader produces
        Stage1Batch samples. Helps catch configuration errors early.
        """
        if self._batch_validated:
            return

        if isinstance(batch, Stage1Batch):
            self._extract_frames_from_stage1_batch(batch)
        else:
            raise TypeError(f"Stage 1 expects Stage1Batch, got {type(batch)}")

        self._batch_validated = True

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Stage1Batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        if isinstance(batch, Stage1Batch):
            frames = self._extract_frames_from_stage1_batch(batch)
            batch_size = self._batch_size_from_stage1_batch(batch)
        else:
            raise TypeError(f"Stage 1 expects Stage1Batch, got {type(batch)}")

        # Forward pass - model returns (loss, metrics_dict)
        #
        # IMPORTANT: Keep the model's notion of "step" aligned with Lightning logging
        # cadence (which is typically (global_step + 1) % N == 0). The LAM model's
        # codebook replacement schedule is step-modulo based, so an off-by-one here
        # can cause replacement stats to never coincide with logged steps.
        loss, metrics = self.model(frames, step=int(self.global_step) + 1)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log(
                "train/loss",
                loss.detach(),
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

            metrics_cfg = self.training_config.metrics
            log_every = int(metrics_cfg.log_every_n_steps)
            step = int(self.global_step) + 1

            if log_every > 0 and step % log_every == 0:
                self.log(
                    "train/lr",
                    self.optimizers().param_groups[0]["lr"],
                    prog_bar=False,
                    batch_size=batch_size,
                )

                # Dynamic logging of model metrics (avoid progress-bar GPU sync)
                for k, v in metrics.items():
                    self.log(
                        f"train/{k}",
                        self._metric_value_for_log(v),
                        prog_bar=False,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

        return loss

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, Stage1Batch):
            batch = move_dataclass_tensors_to_device(batch, device)
            return Stage1Batch(
                image_streams=uint8_image_streams_to_float32(batch.image_streams),
                image_padding_masks=batch.image_padding_masks,
                task_text=batch.task_text,
                subtask_text=batch.subtask_text,
                state=batch.state,
                state_is_pad=batch.state_is_pad,
                action=batch.action,
                action_is_pad=batch.action_is_pad,
                meta=batch.meta,
            )

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Stage1Batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        if isinstance(batch, Stage1Batch):
            frames = self._extract_frames_from_stage1_batch(batch)
            batch_size = self._batch_size_from_stage1_batch(batch)
        else:
            raise TypeError(f"Stage 1 expects Stage1Batch, got {type(batch)}")

        # Forward pass - use step=0 to avoid codebook replacement during validation
        loss, metrics = self.model(frames, step=0)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log(
                "val/loss",
                loss.detach(),
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

            # Dynamic logging of model metrics
            for k, v in metrics.items():
                self.log(
                    f"val/{k}",
                    self._metric_value_for_log(v),
                    sync_dist=True,
                    batch_size=batch_size,
                )

        return loss

    def _resolve_total_training_steps(self) -> int:
        total_steps = int(self.training_config.max_steps)
        if total_steps <= 0:
            raise ValueError("training.max_steps must be > 0 for cosine scheduler")
        return total_steps

    def _resolve_warmup_steps(self, sched_config: DictConfig) -> int:
        warmup_steps = int(sched_config.warmup_steps)
        if warmup_steps < 0:
            raise ValueError("training.scheduler.warmup_steps must be >= 0")
        return warmup_steps

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and LR scheduler.

        Uses LAPA-style optimizer:
        - Separate weight decay groups (2D+ params vs <2D params)
        - AdamW with cosine annealing LR
        - Optional warmup

        Returns:
            Dict with optimizer and lr_scheduler configs
        """
        opt_config = self.training_config.optimizer
        sched_config = self.training_config.scheduler

        # Check for common configuration error: passing a list for LR without multirun
        if isinstance(opt_config.lr, (list, ListConfig)):
            raise ValueError(
                f"Optimizer 'lr' is a list ({opt_config.lr}). "
                "If you intended to run a hyperparameter sweep, make sure to add "
                "'-m' or '--multirun' to your command line arguments."
            )

        # Separate parameters for weight decay
        all_params = list(self.model.parameters())
        wd_params, no_wd_params = separate_weight_decayable_params(all_params)

        # Create optimizer with parameter groups
        optimizer = AdamW(
            [
                {
                    "params": wd_params,
                    "weight_decay": opt_config.weight_decay,
                },
                {
                    "params": no_wd_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=opt_config.lr,
            betas=tuple(opt_config.betas),
            eps=opt_config.eps,
        )

        # Create LR scheduler (optional)
        if sched_config.type == "none":
            # No scheduler - return optimizer only
            return optimizer
        elif sched_config.type == "cosine":
            total_steps = self._resolve_total_training_steps()
            warmup_steps = self._resolve_warmup_steps(sched_config)
            warmup_steps = min(warmup_steps, total_steps)

            min_lr = float(sched_config.min_lr)
            base_lr = float(opt_config.lr)
            warmup_start_lr = float(sched_config.warmup_start_lr)

            if warmup_steps > 0:
                if base_lr <= 0:
                    raise ValueError(
                        "training.optimizer.lr must be > 0 when warmup is enabled"
                    )
                start_factor = warmup_start_lr / base_lr
                if start_factor <= 0:
                    raise ValueError("training.scheduler.warmup_start_lr must be > 0")

                warmup = LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_steps = max(1, total_steps - warmup_steps)
                cosine = CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_steps,
                    eta_min=min_lr,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_steps),
                    eta_min=min_lr,
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr",
                },
            }
        else:
            raise NotImplementedError(
                f"Scheduler type '{sched_config.type}' not implemented"
            )

    def encode_latents(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode frame pairs to get latent actions and codebook indices.

        Args:
            batch: Frame pairs [B, C, 2, H, W]

        Returns:
            (latent_actions, codebook_indices)
            latent_actions: [B, code_seq_len, dim] projected to transformer dim
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            # Get codebook indices [B, code_seq_len]
            indices = self.model(batch, return_only_codebook_ids=True)
            # Get raw codebook vectors [B, code_seq_len, quant_dim]
            raw_latents = self.model.vq.codebooks[indices]
            # Project from quant_dim to transformer dim [B, code_seq_len, dim]
            latents = self.model.vq.project_out(raw_latents)
        self.train(was_training)
        return latents, indices

    def decode_with_latents(
        self,
        first_frames: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Decode first frames with given latent actions.

        This enables latent transfer: apply action from one pair to another scene.
        Returns None if no reconstruction decoder is enabled.

        Args:
            first_frames: First frames [B, C, 1, H, W] or [B, C, H, W]
            latent_actions: Latent actions from encoder

        Returns:
            Reconstructed next frames [B, C, 1, H, W], or None if no
                reconstruction decoder is enabled
        """
        import math
        from einops import rearrange

        was_training = self.training
        self.eval()
        with torch.no_grad():
            first_frames = first_frames.to(self.device)
            latent_actions = latent_actions.to(self.device)

            # Ensure correct shape
            if first_frames.ndim == 4:
                first_frames = first_frames.unsqueeze(
                    2
                )  # [B, C, H, W] -> [B, C, 1, H, W]

            # Get first frame tokens
            # Use decoder_context_projection to match model forward pass (handles DINO vs learned embeddings)
            first_frame_tokens = self.model.decoder_context_projection(first_frames)

            # Reshape latents for decode
            code_seq_len = self.model.code_seq_len
            if math.sqrt(code_seq_len) % 1 == 0:
                action_h = int(math.sqrt(code_seq_len))
                action_w = int(math.sqrt(code_seq_len))
            elif code_seq_len == 2:
                action_h, action_w = 2, 1
            else:
                action_h, action_w = code_seq_len, 1

            # Reshape latents: [B, seq, dim] -> [B, t, h, w, d]
            if latent_actions.ndim == 2:
                latent_actions = latent_actions.unsqueeze(1)  # [B, dim] -> [B, 1, dim]
            latent_actions = rearrange(
                latent_actions, "b (t h w) d -> b t h w d", t=1, h=action_h, w=action_w
            )

            # Decode
            recon = self.model.decode(first_frame_tokens, latent_actions)
        self.train(was_training)
        return recon
