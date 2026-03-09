"""
Visualization strategies for LAM validation.
"""

from typing import Any, Dict, List, Optional

import torch
from torchvision.utils import make_grid
from einops import rearrange
import lightning.pytorch as pl

from .core import ValidationStrategy, ValidationCache


class BasicVisualizationStrategy(ValidationStrategy):
    """
    Basic reconstruction visualization.

    Shows:
    - Fixed samples: diverse samples across datasets, same every validation
    - Random samples: different samples each time (diversity check)
    - Per-bucket samples: separate grids for each configured bucket (via filters)
    - Training samples: reconstructions from training data

    Bucket filters are configured in `validation.buckets` and are passed
    to this strategy via `bucket_filters` (bucket_name -> filter dict).
    """

    def __init__(
        self,
        name: str = "basic_visualization",
        enabled: bool = True,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        num_train_samples: int = 8,
        visualize_train: bool = True,
        visualize_val: bool = True,
        visualize_per_bucket: bool = True,
        samples_per_bucket: int = 4,
        bucket_filters: Optional[Dict[str, Dict[str, Any]]] = None,
        every_n_validations: int = 1,  # Default: always run
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.num_train_samples = num_train_samples
        self.visualize_train = visualize_train
        self.visualize_val = visualize_val
        self.visualize_per_bucket = visualize_per_bucket
        self.samples_per_bucket = samples_per_bucket
        self.bucket_filters = bucket_filters
        self._warned_missing_train_preview_buffer = False

    def needs_caching(self) -> bool:
        return True  # Need frames for visualization

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate reconstruction visualizations for both train and val."""
        metrics = {}
        wandb_logger = self._get_wandb_logger(trainer)
        produced = 0

        # Use bucket name for prefixing if available
        bucket_name = cache.bucket_name or ""
        prefix = f"val/{bucket_name}" if bucket_name else "val"

        # === Training samples visualization ===
        if self.visualize_train and not bucket_name:  # Only for global cache
            produced += self._visualize_training_samples(
                cache, pl_module, trainer, wandb_logger
            )

        # === Validation samples visualization ===
        if self.visualize_val:
            produced += self._visualize_validation_samples(
                cache, pl_module, trainer, wandb_logger, prefix
            )

        if produced <= 0:
            if wandb_logger is None:
                return self.no_output("wandb_logger_unavailable")
            return self.no_output("no_visualizations_rendered")
        return self.success(produced=produced, metrics=metrics)

    def _visualize_training_samples(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
    ) -> int:
        """Visualize training samples from non-intrusive train preview buffer."""
        if wandb_logger is None:
            return 0
        produced = 0

        train_frames, train_metadata = self._sample_from_train_preview_buffer(
            trainer,
            self.num_train_samples * 2,  # Sample extra for both fixed and random
        )

        if train_frames is None or len(train_frames) == 0:
            return produced

        # === Fixed training samples (same across validations for progress tracking) ===
        if cache.train_frames is None or len(cache.train_frames) == 0:
            # First validation: cache a fixed set of training samples
            cache.train_frames = train_frames[: self.num_train_samples]
            cache.train_metadata = (
                train_metadata[: self.num_train_samples] if train_metadata else None
            )

        fixed_grid = self._create_recon_grid(cache.train_frames, pl_module)
        if fixed_grid is not None:
            wandb_logger.log_image(
                key="train/fixed_reconstructions",
                images=[fixed_grid],
                caption=[f"Step {trainer.global_step} (fixed training samples)"],
            )
            produced += 1

        # === Random training samples (different each validation for diversity) ===
        random_grid = self._create_recon_grid(
            train_frames[: self.num_train_samples], pl_module
        )
        if random_grid is not None:
            wandb_logger.log_image(
                key="train/random_reconstructions",
                images=[random_grid],
                caption=[f"Step {trainer.global_step} (random training samples)"],
            )
            produced += 1

        # === Per-bucket training visualization ===
        if self.visualize_per_bucket and train_metadata:
            produced += self._visualize_buckets(
                train_frames,
                train_metadata,
                cache,
                pl_module,
                trainer,
                wandb_logger,
                prefix="train",
            )
        return produced

    def _visualize_validation_samples(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
        prefix: str = "val",
    ) -> int:
        """Visualize validation samples from cache."""
        produced = 0
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()

        if all_frames is None or len(all_frames) == 0:
            return produced

        # Log cache distribution for debugging
        distribution = cache.get_dataset_distribution()
        bucket_name = cache.bucket_name or "global"
        if distribution:
            print(
                f"  [{bucket_name}] Cached validation samples per datasource: {distribution}"
            )
            print(
                f"  [{bucket_name}] Total cached validation samples: {sum(distribution.values())}"
            )

        # === Fixed samples (diverse across datasets) ===
        if cache.fixed_frames is not None and len(cache.fixed_frames) > 0:
            fixed_grid = self._create_recon_grid(cache.fixed_frames, pl_module)
            if wandb_logger and fixed_grid is not None:
                wandb_logger.log_image(
                    key=f"{prefix}/fixed_reconstructions",
                    images=[fixed_grid],
                    caption=[f"Step {trainer.global_step} (fixed diverse samples)"],
                )
                produced += 1

        # === Random samples (different each time) ===
        n_random = min(self.num_random_samples, len(all_frames))
        if n_random > 0 and wandb_logger:
            random_indices = torch.randperm(len(all_frames))[:n_random]
            random_frames = all_frames[random_indices]
            random_grid = self._create_recon_grid(random_frames, pl_module)
            if random_grid is not None:
                wandb_logger.log_image(
                    key=f"{prefix}/random_reconstructions",
                    images=[random_grid],
                    caption=[f"Step {trainer.global_step} (random samples)"],
                )
                produced += 1

        # === Per-bucket visualization ===
        if self.visualize_per_bucket and all_metadata:
            produced += self._visualize_buckets(
                all_frames,
                all_metadata,
                cache,
                pl_module,
                trainer,
                wandb_logger,
                prefix="val",
            )
        return produced

    def _visualize_buckets(
        self,
        all_frames: torch.Tensor,
        all_metadata: List[Dict[str, Any]],
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        wandb_logger,
        prefix: str = "val",
    ) -> int:
        """Visualize samples grouped by buckets using side dataloaders or cache."""
        if wandb_logger is None:
            return 0
        produced = 0

        # Check if DataModule supports side dataloaders
        datamodule = getattr(trainer, "datamodule", None)
        use_dataloaders = datamodule is not None

        if not self.bucket_filters:
            return produced

        for bucket_name, filters in self.bucket_filters.items():
            bucket_frames = None

            # Try to get data from side dataloader first (Targeted Evaluation)
            if use_dataloaders:
                try:
                    # Determine if we need train or val loader
                    if prefix == "train" and hasattr(
                        datamodule, "train_bucket_dataloader"
                    ):
                        loader = datamodule.train_bucket_dataloader(bucket_name)
                    elif prefix == "val" and hasattr(
                        datamodule, "val_bucket_dataloader"
                    ):
                        loader = datamodule.val_bucket_dataloader(bucket_name)
                    else:
                        loader = None

                    if loader:
                        # Fetch one batch
                        batch = next(iter(loader))
                        if isinstance(batch, dict):
                            bucket_frames = batch["frames"]
                        else:
                            bucket_frames = batch
                        # Move to device/cpu as needed (viz expects CPU usually)
                        bucket_frames = (
                            bucket_frames[: self.samples_per_bucket].detach().cpu()
                        )
                except (ValueError, StopIteration, NotImplementedError):
                    # Dataloader might not exist for this bucket or be empty
                    pass

            # Fallback to cache filtering if dataloader failed
            if bucket_frames is None:
                bucket_frames, _ = cache.get_frames_by_filter(
                    filters, frames=all_frames, metadata=all_metadata
                )

            if bucket_frames is not None and len(bucket_frames) > 0:
                # Randomly sample if we have more than needed (and came from cache)
                if len(bucket_frames) > self.samples_per_bucket:
                    indices = torch.randperm(len(bucket_frames))[
                        : self.samples_per_bucket
                    ]
                    samples = bucket_frames[indices]
                else:
                    samples = bucket_frames

                bucket_grid = self._create_recon_grid(samples, pl_module)
                if bucket_grid is not None:
                    wandb_logger.log_image(
                        key=f"{prefix}/reconstructions_{bucket_name}",
                        images=[bucket_grid],
                        caption=[f"Step {trainer.global_step} ({bucket_name})"],
                    )
                    produced += 1
        return produced

    def _sample_from_train_preview_buffer(
        self,
        trainer: pl.Trainer,
        num_samples: int,
    ) -> tuple[Optional[torch.Tensor], Optional[List[Dict[str, Any]]]]:
        """Sample frames from TrainPreviewBufferCallback without touching train dataloader."""
        try:
            from lam.callbacks import TrainPreviewBufferCallback
        except Exception as e:
            if not self._warned_missing_train_preview_buffer:
                print(f"Warning: could not import TrainPreviewBufferCallback: {e}")
                self._warned_missing_train_preview_buffer = True
            return None, None

        callbacks = list(getattr(trainer, "callbacks", []))
        for callback in callbacks:
            if isinstance(callback, TrainPreviewBufferCallback):
                return callback.sample(num_samples)

        if not self._warned_missing_train_preview_buffer:
            print(
                "Warning: visualize_train is enabled but TrainPreviewBufferCallback "
                "is not registered; skipping train visualizations."
            )
            self._warned_missing_train_preview_buffer = True
        return None, None

    def _create_recon_grid(
        self,
        frames: torch.Tensor,
        pl_module: pl.LightningModule,
    ) -> Optional[torch.Tensor]:
        """Create reconstruction grid.

        Returns:
            - [frame_t, frame_t+offset, reconstruction] grid when aux decoder is available
            - [frame_t, frame_t+offset] grid fallback otherwise
        """
        if len(frames) == 0:
            return None

        # Create a lightweight fallback when aux decoder is disabled.
        # This keeps sample visualization available without reconstruction overhead.
        has_aux_decoder = (
            getattr(getattr(pl_module, "model", None), "aux_decoder", None) is not None
        )
        recons = None
        if has_aux_decoder:
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                recons = pl_module.model(
                    frames.to(pl_module.device),
                    return_recons_only=True,
                )
            pl_module.train(was_training)

        # Create grid: [frame_t, frame_t+offset, reconstruction?]
        frame_t = frames[:, :, 0].cpu()
        frame_t_plus = frames[:, :, 1].cpu()
        if recons is None:
            imgs = torch.stack([frame_t, frame_t_plus], dim=0)
            nrow = 2
        else:
            recons = recons.cpu()
            imgs = torch.stack([frame_t, frame_t_plus, recons], dim=0)
            nrow = 3
        imgs = rearrange(imgs, "r b c h w -> (b r) c h w")
        imgs = imgs.clamp(0.0, 1.0)

        return make_grid(imgs, nrow=nrow, normalize=False)
