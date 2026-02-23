"""
PyTorch Lightning task wrapper for LAQ training.

Wraps LatentActionQuantization in a LightningModule with:
- LAPA-style optimizer (separate weight decay groups)
- Loss logging and codebook usage tracking
- Hydra configuration integration
- Optional EMA
"""

from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig, OmegaConf

from laq.models.latent_action_quantization import LatentActionQuantization, DinoConfig
from laq.models.flow import FlowConfig


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


class LAQTask(pl.LightningModule):
    """
    PyTorch Lightning task for LAQ training.

    Wraps LatentActionQuantization model with training logic matching LAPA:
    - AdamW optimizer with separated weight decay groups
    - Cosine annealing LR scheduler with warmup
    - Reconstruction loss + codebook usage tracking
    - Visualization support via callbacks

    Args:
        model_config: Model configuration (DictConfig or dict)
        training_config: Training configuration (DictConfig or dict)
        use_ema: Whether to use EMA (handled via callback if True)
    """

    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
        use_ema: bool = False,
    ):
        super().__init__()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Store configs
        self.model_config = model_config
        self.training_config = training_config
        self.use_ema = use_ema

        # Build flow config if specified
        flow_config = None
        if "flow" in model_config and model_config.flow is not None:
            flow_cfg = model_config.flow
            flow_enabled = bool(flow_cfg.get("enabled", True))
            if flow_enabled:
                try:
                    flow_config = FlowConfig(
                        model=flow_cfg.model,
                        loss_weight=flow_cfg.loss_weight,
                        decoder_depth=flow_cfg.decoder_depth,
                        warmup_steps=flow_cfg.get("warmup_steps", 0),
                        teacher_num_flow_updates=flow_cfg.get("teacher_num_flow_updates", 12),
                        teacher_chunk_size=flow_cfg.get("teacher_chunk_size", 64),
                        summary_loss_weight=flow_cfg.get("summary_loss_weight", 0.0),
                        summary_static_eps=flow_cfg.get("summary_static_eps", 1e-6),
                    )
                except Exception as exc:
                    raise ValueError(
                        "Invalid flow config. Expected:\n"
                        "  model.flow: null  # to disable\n"
                        "  # or\n"
                        "  model.flow:\n"
                        "    enabled: true\n"
                        "    model: raft_small|raft_large\n"
                        "    loss_weight: <float>\n"
                        "    decoder_depth: <int>\n"
                        "    warmup_steps: <int, optional>\n"
                        "    teacher_num_flow_updates: <int, optional>\n"
                        "    teacher_chunk_size: <int, optional>\n"
                        "    summary_loss_weight: <float, optional>\n"
                        "    summary_static_eps: <float, optional>\n"
                    ) from exc
        dino_config = None
        if "dino" in model_config and model_config.dino is not None:
            dino_cfg = model_config.dino
            dino_enabled = bool(dino_cfg.get("enabled", True))
            if dino_enabled:
                try:
                    dino_config = DinoConfig(
                        loss_weight=float(dino_cfg.get("loss_weight", 1.0)),
                        warmup_steps=int(dino_cfg.get("warmup_steps", 0)),
                    )
                except Exception as exc:
                    raise ValueError(
                        "Invalid dino config. Expected:\n"
                        "  model.dino:\n"
                        "    enabled: true|false\n"
                        "    loss_weight: <float, optional>\n"
                        "    warmup_steps: <int, optional>\n"
                    ) from exc

        # Build codebook replacement schedule if specified
        codebook_replace_schedule = None
        if "codebook_replace_schedule" in model_config and model_config.codebook_replace_schedule is not None:
            # Convert from list of lists to list of tuples
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

        metrics_cfg = training_config.get("metrics")
        if metrics_cfg is None or metrics_cfg.get("num_unique_codes_every_n_steps") is None:
            raise ValueError(
                "Missing `training.metrics.num_unique_codes_every_n_steps` config. Expected:\n"
                "training:\n"
                "  metrics:\n"
                "    log_every_n_steps: <int>\n"
                "    num_unique_codes_every_n_steps: <int>\n"
            )

        # Initialize LAQ model
        self.model = LatentActionQuantization(
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
            vq_discarding_threshold=model_config.get("vq_discarding_threshold", 0.1),
            vq_discarding_threshold_schedule=vq_discarding_threshold_schedule,
            channels=model_config.get("channels", 3),
            attn_dropout=model_config.get("attn_dropout", 0.0),
            ff_dropout=model_config.get("ff_dropout", 0.0),
            latent_ablation=model_config.get("latent_ablation", "none"),
            use_dinov3_encoder=model_config.get("use_dinov3_encoder", False),
            dinov3_model_name=model_config.get("dinov3_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            dinov3_pool_to_grid=model_config.get("dinov3_pool_to_grid", None),
            metrics_num_unique_codes_every_n_steps=int(metrics_cfg.num_unique_codes_every_n_steps),
            dino_config=dino_config,
            # Training decoder flags
            use_pixel_decoder=model_config.get("use_pixel_decoder", False),
            # Interpretability decoder flag
            use_aux_decoder=model_config.get("use_aux_decoder", True),
            flow_config=flow_config,
            codebook_replace_schedule=codebook_replace_schedule,
        )

        # Storage for validation and training batches (for visualization)
        self.validation_batch = None
        self.training_batch = None

        # Flag for one-time batch validation (to catch interface issues early)
        self._batch_validated = False

    def forward(
        self,
        video: torch.Tensor,
        step: int = 0,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ) -> Any:
        """
        Forward pass through LAQ model.

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
        batches with standardized keys (frames, episode_id, frame_idx, etc.).
        Helps catch configuration errors (e.g., wrong collate function) early.

        Validates against STANDARD_BATCH_KEYS to ensure interface parity between
        LAQDataModule and OXEDataModule.
        """
        if self._batch_validated:
            return

        # Only validate when batch is a dict (metadata mode enabled)
        if isinstance(batch, dict):
            from common.data import validate_batch_keys, STANDARD_BATCH_KEYS
            import logging

            logger = logging.getLogger(__name__)

            # Validate all standard keys to ensure interface parity
            # This catches misconfigured dataloaders early
            validate_batch_keys(
                batch,
                required_keys=list(STANDARD_BATCH_KEYS),
                raise_on_missing=True,
            )

            # Log what keys we got (helpful for debugging)
            if self.trainer.is_global_zero:
                logger.info(f"Batch keys validated: {list(batch.keys())}")
                logger.info(f"Required standard keys present: {list(STANDARD_BATCH_KEYS)}")

        self._batch_validated = True

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Frame pairs [B, C, 2, H, W] or metadata dict
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Handle metadata dict if present
        if isinstance(batch, dict):
            frames = batch["frames"]
        else:
            frames = batch

        # Forward pass - model returns (loss, metrics_dict)
        #
        # IMPORTANT: Keep the model's notion of "step" aligned with Lightning logging
        # cadence (which is typically (global_step + 1) % N == 0). The LAQ model's
        # codebook replacement schedule is step-modulo based, so an off-by-one here
        # can cause replacement stats to never coincide with logged steps.
        loss, metrics = self.model(frames, step=int(self.global_step) + 1)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)

            metrics_cfg = self.training_config.metrics
            log_every = int(metrics_cfg.log_every_n_steps)
            step = int(self.global_step) + 1

            if log_every > 0 and step % log_every == 0:
                self.log(
                    "train/lr",
                    self.optimizers().param_groups[0]["lr"],
                    prog_bar=False,
                )

                # Dynamic logging of model metrics (avoid progress-bar GPU sync)
                for k, v in metrics.items():
                    self.log(f"train/{k}", v, prog_bar=False, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.training_batch is None:
            self.training_batch = frames[:8].detach().cpu()

        return loss

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        if isinstance(batch, dict):
            frames = batch.get("frames")
            if isinstance(frames, torch.Tensor) and frames.dtype == torch.uint8:
                batch["frames"] = frames.to(dtype=torch.float32).div_(255.0)
            return batch

        if isinstance(batch, torch.Tensor) and batch.dtype == torch.uint8:
            return batch.to(dtype=torch.float32).div_(255.0)

        return batch

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Frame pairs [B, C, 2, H, W] or metadata dict
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Handle metadata dict if present
        if isinstance(batch, dict):
            frames = batch["frames"]
        else:
            frames = batch

        # Forward pass - use step=0 to avoid codebook replacement during validation
        loss, metrics = self.model(frames, step=0)

        # Log metrics (skip if no trainer attached, e.g., in unit tests)
        if self._trainer is not None:
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            
            # Dynamic logging of model metrics
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)

        # Store first batch for visualization
        if batch_idx == 0 and self.validation_batch is None:
            self.validation_batch = frames[:8].detach().cpu()

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset training batch storage
        self.training_batch = None

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Reset validation batch storage
        self.validation_batch = None

    def _resolve_total_training_steps(self, sched_config: DictConfig) -> Optional[int]:
        """
        Resolve total optimizer steps for step-based LR scheduling.

        Priority:
        1) Explicit `training_config.max_steps` when set (>0)
        2) Trainer `max_steps` when set (>0)
        3) Trainer `estimated_stepping_batches` when finite
        """
        # 1) Training config (Hydra) override
        cfg_max_steps = self.training_config.get("max_steps")
        if cfg_max_steps is not None:
            try:
                cfg_max_steps_int = int(cfg_max_steps)
                if cfg_max_steps_int > 0:
                    return cfg_max_steps_int
            except (TypeError, ValueError):
                pass

        # 2/3) Trainer-derived values (preferred when available)
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            trainer_max_steps = getattr(trainer, "max_steps", None)
            if trainer_max_steps not in (None, -1):
                try:
                    trainer_max_steps_int = int(trainer_max_steps)
                    if trainer_max_steps_int > 0:
                        return trainer_max_steps_int
                except (TypeError, ValueError):
                    pass

            est = getattr(trainer, "estimated_stepping_batches", None)
            if est is not None:
                try:
                    est_int = int(est)
                    if est_int > 0:
                        return est_int
                except (TypeError, ValueError, OverflowError):
                    pass

            # Fallback if estimated_stepping_batches isn't available yet.
            num_batches = getattr(trainer, "num_training_batches", None)
            max_epochs = getattr(trainer, "max_epochs", None)
            accumulate = getattr(trainer, "accumulate_grad_batches", 1)
            if (
                num_batches not in (None, float("inf"))
                and max_epochs not in (None, -1)
            ):
                try:
                    num_batches_int = int(num_batches)
                    max_epochs_int = int(max_epochs)
                    accumulate_int = int(accumulate) if int(accumulate) > 0 else 1
                    if num_batches_int > 0 and max_epochs_int > 0:
                        steps_per_epoch = math.ceil(num_batches_int / accumulate_int)
                        return steps_per_epoch * max_epochs_int
                except (TypeError, ValueError, OverflowError):
                    pass

        return None

    def _resolve_warmup_steps(self, sched_config: DictConfig) -> int:
        """
        Resolve warmup steps.
        """
        warmup_steps = sched_config.get("warmup_steps", None)
        if warmup_steps is not None:
            try:
                return max(0, int(warmup_steps))
            except (TypeError, ValueError):
                return 0
        return 0

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
        if sched_config.get("type") == "none" or sched_config.get("type") is None:
            # No scheduler - return optimizer only
            return optimizer
        elif sched_config.type == "cosine":
            total_steps = self._resolve_total_training_steps(sched_config)
            if total_steps is None:
                raise ValueError(
                    "LAQ cosine scheduler requires total training steps. "
                    "Set `training.max_steps` (recommended for streaming/variable-length epochs)."
                )

            warmup_steps = self._resolve_warmup_steps(sched_config)
            warmup_steps = min(warmup_steps, total_steps)

            min_lr = float(sched_config.get("min_lr", 0.0) or 0.0)
            base_lr = float(opt_config.lr)
            warmup_start_lr = float(sched_config.get("warmup_start_lr", min_lr) or min_lr)

            if warmup_steps > 0:
                start_factor = warmup_start_lr / base_lr if base_lr > 0 else 1.0
                # LinearLR expects multiplicative factor; clamp to avoid negative/zero.
                start_factor = max(1e-8, start_factor)
                
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
            raise NotImplementedError(f"Scheduler type '{sched_config.type}' not implemented")

    def get_validation_batch(self) -> Optional[torch.Tensor]:
        """
        Get stored validation batch for visualization.

        Returns:
            Validation batch tensor or None
        """
        return self.validation_batch

    def get_training_batch(self) -> Optional[torch.Tensor]:
        """
        Get stored training batch for visualization.

        Returns:
            Training batch tensor or None
        """
        return self.training_batch

    def generate_reconstructions(
        self,
        batch: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Generate reconstructions for visualization.

        Args:
            batch: Frame pairs [B, C, 2, H, W]

        Returns:
            Reconstructions [B, C, H, W], or None if aux_decoder is disabled
        """
        self.eval()
        with torch.no_grad():
            recons = self.model(batch.to(self.device), return_recons_only=True)
        self.train()
        return recons

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
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            # Get codebook indices [B, code_seq_len]
            indices = self.model(batch, return_only_codebook_ids=True)
            # Get raw codebook vectors [B, code_seq_len, quant_dim]
            raw_latents = self.model.vq.codebooks[indices]
            # Project from quant_dim to transformer dim [B, code_seq_len, dim]
            latents = self.model.vq.project_out(raw_latents)
        self.train()
        return latents, indices

    def decode_with_latents(
        self,
        first_frames: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Decode first frames with given latent actions.

        This enables latent transfer: apply action from one pair to another scene.
        Returns None if aux_decoder is disabled.

        Args:
            first_frames: First frames [B, C, 1, H, W] or [B, C, H, W]
            latent_actions: Latent actions from encoder

        Returns:
            Reconstructed next frames [B, C, 1, H, W], or None if aux_decoder disabled
        """
        import math
        from einops import rearrange

        self.eval()
        with torch.no_grad():
            first_frames = first_frames.to(self.device)
            latent_actions = latent_actions.to(self.device)

            # Ensure correct shape
            if first_frames.ndim == 4:
                first_frames = first_frames.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

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
                latent_actions, 'b (t h w) d -> b t h w d',
                t=1, h=action_h, w=action_w
            )

            # Decode
            recon = self.model.decode(first_frame_tokens, latent_actions)

        self.train()
        return recon
