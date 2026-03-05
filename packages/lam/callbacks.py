"""
PyTorch Lightning callbacks for Stage-1 LAM training.

Includes:
- ValidationStrategyCallback: Flexible validation with bucket-aware routing
- EMACallback: Exponential moving average of model weights
- TrainPreviewBufferCallback: Non-intrusive train sample snapshots for visualization
"""

import gc
import logging
from collections import deque
from typing import Optional, List, Dict, Any, Deque

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from common.batch_utils import select_primary_image_stream, temporal_frames_to_bcthw
from common.lerobot_v3_types import Stage1Batch

logger = logging.getLogger("stage1.training")


def _stage1_batch_to_frames(batch: Stage1Batch) -> torch.Tensor:
    return temporal_frames_to_bcthw(
        select_primary_image_stream(batch.image_streams),
        expected_time_steps=2,
    )


def _stage1_batch_to_metadata(batch: Stage1Batch) -> List[Dict[str, Any]]:
    frames = _stage1_batch_to_frames(batch)
    batch_size = int(frames.shape[0])
    meta_dict = dict(batch.meta or {})
    task_text = list(batch.task_text or [])
    subtask_text = list(batch.subtask_text or [])
    out: List[Dict[str, Any]] = []
    for i in range(batch_size):
        meta: Dict[str, Any] = {}
        for key, value in meta_dict.items():
            if isinstance(value, (list, tuple)):
                if i < len(value):
                    meta[key] = value[i]
            else:
                meta[key] = value
        if i < len(task_text):
            meta["language"] = task_text[i]
            meta.setdefault("task", task_text[i])
        if i < len(subtask_text):
            meta["subtask"] = subtask_text[i]
        out.append(meta)
    return out


class TrainPreviewBufferCallback(Callback):
    """
    Cache a small rolling window of train samples for visualization.

    This callback records snapshots from already-consumed train batches in
    `on_train_batch_end`, so validation strategies can visualize train samples
    without creating a new iterator over `trainer.train_dataloader`.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_samples: int = 256,
        samples_per_batch: int = 4,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.max_samples = int(max_samples)
        self.samples_per_batch = int(samples_per_batch)
        if self.max_samples <= 0:
            raise ValueError("TrainPreviewBufferCallback.max_samples must be > 0")
        if self.samples_per_batch <= 0:
            raise ValueError("TrainPreviewBufferCallback.samples_per_batch must be > 0")
        if metadata_keys is None:
            metadata_keys = [
                "dataset_name",
                "dataset_short",
                "language",
                "task",
                "environment",
                "scene_id",
            ]
        self.metadata_keys = [str(k) for k in metadata_keys]
        self._frames: Deque[torch.Tensor] = deque(maxlen=self.max_samples)
        self._metadata: Deque[Dict[str, Any]] = deque(maxlen=self.max_samples)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.enabled:
            return
        if isinstance(batch, Stage1Batch):
            frames = _stage1_batch_to_frames(batch)
            metadata_list = _stage1_batch_to_metadata(batch)
        elif isinstance(batch, dict):
            frames = batch.get("frames")
            metadata_list = None
        else:
            return
        if not isinstance(frames, torch.Tensor) or frames.ndim == 0:
            return
        if metadata_list is None:
            metadata_list = [self._extract_metadata(batch, i) for i in range(int(frames.shape[0]))]

        batch_size = int(frames.shape[0])
        take = min(self.samples_per_batch, batch_size)
        if take <= 0:
            return

        for i in range(take):
            frame = frames[i].detach().cpu()
            if frame.dtype == torch.uint8:
                frame = frame.to(dtype=torch.float32).div_(255.0)
            self._frames.append(frame)
            self._metadata.append(metadata_list[i])

    def sample(
        self,
        num_samples: int,
    ) -> tuple[Optional[torch.Tensor], Optional[List[Dict[str, Any]]]]:
        if num_samples <= 0:
            raise ValueError("TrainPreviewBufferCallback.sample(num_samples) expects > 0")

        available = len(self._frames)
        if available == 0:
            return None, None

        take = min(int(num_samples), available)
        selected = torch.randperm(available)[:take].tolist()
        frames = torch.stack([self._frames[i] for i in selected], dim=0)
        metadata = [self._metadata[i] for i in selected]
        return frames, metadata

    def _extract_metadata(self, batch: Dict[str, Any], i: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        for key in self.metadata_keys:
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, (list, tuple)):
                if i < len(value):
                    meta[key] = value[i]
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    meta[key] = value.item()
                elif i < value.shape[0]:
                    item = value[i]
                    if isinstance(item, torch.Tensor) and item.ndim == 0:
                        meta[key] = item.item()
            else:
                meta[key] = value
        return meta


class ValidationStrategyCallback(Callback):
    """
    Flexible validation callback with bucket-aware routing.

    Architecture (Composition Pattern):
    - Buckets: Named data subsets with filters (e.g., "language_table", "bridge")
    - Strategies: Self-contained validation logic with embedded bucket bindings

    Features:
    - Per-bucket caching: Each bucket has its own cache
    - Strategy-embedded binding: Strategies read from their own `buckets` property
    - Automatic applicability checks: Strategies check if they have enough valid data

    Args:
        strategies: List of ValidationStrategy instances (with buckets property)
        bucket_configs: Dict of bucket name -> BucketConfig or dict with filters
        num_fixed_samples: Number of fixed samples per bucket
        max_cached_samples: Maximum samples for global cache (fallback)
    """

    def __init__(
        self,
        strategies: Optional[List] = None,
        bucket_configs: Optional[Dict[str, Any]] = None,
        num_fixed_samples: int = 8,
        num_random_samples: int = 8,
        max_cached_samples: int = 256,
        run_gc_after_validation: bool = True,
    ):
        super().__init__()
        self.strategies = strategies or []
        self.num_fixed_samples = num_fixed_samples
        self.num_random_samples = num_random_samples
        self.max_cached_samples = max_cached_samples
        self.run_gc_after_validation = run_gc_after_validation

        # Import here to avoid circular imports
        from lam.validation import ValidationCache, BucketConfig

        # Create bucket configs
        self.bucket_configs: Dict[str, BucketConfig] = {}
        if bucket_configs:
            for name, cfg in bucket_configs.items():
                if isinstance(cfg, BucketConfig):
                    self.bucket_configs[name] = cfg
                else:
                    self.bucket_configs[name] = BucketConfig(
                        name=name,
                        filters=cfg.get("filters", {}),
                        max_samples=cfg.get("max_samples", 100),
                        is_holdout=cfg.get("is_holdout", False),
                    )

        # Create per-bucket caches
        self.bucket_caches: Dict[str, ValidationCache] = {}
        for name, cfg in self.bucket_configs.items():
            cache = ValidationCache()
            cache.bucket_name = name
            cache.is_holdout = cfg.is_holdout
            cache.max_samples = cfg.max_samples
            self.bucket_caches[name] = cache

        # Global cache (fallback for strategies without bucket bindings)
        self.global_cache = ValidationCache()
        self.global_cache.max_samples = max_cached_samples

        # Track fixed sample indices
        self.fixed_indices: Optional[List[int]] = None
        self.validation_count = 0
        self._first_full_validation_done = False

    def _any_strategy_needs_codes(self) -> bool:
        """Check if any strategy will need codebook indices on this validation.

        Note: This is called during batch processing, before validation_count is
        incremented. We need to check with (count + 1) to predict what strategies
        will run at epoch end.
        """
        for strategy in self.strategies:
            if strategy.will_run_next() and strategy.needs_codes():
                return True
        return False

    def _extract_metadata(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-sample metadata from batch."""
        frames = batch["frames"]
        batch_metadata = []
        batch_size = frames.shape[0]

        for i in range(batch_size):
            meta = {}
            for key in batch.keys():
                if key == "frames":
                    continue
                val = batch[key]

                # Special handling for 'action' and 'initial_state' which get transposed
                if (key == "action" or key == "initial_state") and isinstance(val, (list, tuple)) and len(val) > 0:
                    if isinstance(val[0], torch.Tensor) and val[0].ndim > 0:
                        dims = [v[i].item() for v in val if i < len(v)]
                        meta[key] = dims
                    else:
                        meta[key] = val[i] if i < len(val) else None
                elif isinstance(val, (list, tuple)):
                    meta[key] = val[i] if i < len(val) else None
                elif isinstance(val, torch.Tensor):
                    if val.ndim > 0 and i < len(val):
                        meta[key] = val[i].item() if val[i].ndim == 0 else val[i].tolist()
                    elif val.ndim == 0:
                        meta[key] = val.item()
                else:
                    meta[key] = val
            batch_metadata.append(meta)

        return batch_metadata

    def _select_diverse_fixed_samples(
        self,
        cache,
        num_samples: int,
    ) -> None:
        """Select diverse fixed samples for a cache."""
        all_frames = cache.get_all_frames()
        all_metadata = cache.get_all_metadata()

        if all_frames is None or len(all_frames) < num_samples:
            return

        # Group by dataset identity for diversity.
        by_dataset: Dict[str, List[int]] = {}
        for i, meta in enumerate(all_metadata):
            dtype = meta.get("dataset_short", "unknown")
            if dtype not in by_dataset:
                by_dataset[dtype] = []
            by_dataset[dtype].append(i)

        selected_indices = []
        dataset_types = list(by_dataset.keys())
        samples_per_dataset = max(1, num_samples // len(dataset_types)) if dataset_types else num_samples

        for dtype in dataset_types:
            indices = by_dataset[dtype]
            shuffled = torch.randperm(len(indices)).tolist()
            for j in shuffled[:samples_per_dataset]:
                if len(selected_indices) < num_samples:
                    selected_indices.append(indices[j])

        # Fill remaining with random
        remaining = [i for i in range(len(all_frames)) if i not in selected_indices]
        while len(selected_indices) < num_samples and remaining:
            idx = remaining.pop(torch.randint(len(remaining), (1,)).item())
            selected_indices.append(idx)

        cache.fixed_indices = selected_indices
        cache.fixed_frames = all_frames[selected_indices]
        cache.fixed_metadata = [all_metadata[i] for i in selected_indices]

    @staticmethod
    def _result_is_no_output(result: Any) -> tuple[bool, str]:
        if not isinstance(result, dict):
            raise TypeError("Validation strategy must return dict result")
        if "_produced" not in result:
            raise KeyError("Validation strategy result missing required '_produced' field")
        produced = int(result.get("_produced", 0))
        if produced > 0:
            return False, ""
        reason = result.get("_reason")
        return True, str(reason) if reason is not None else "no outputs produced"

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Clear all caches at start of validation."""
        self.global_cache.clear()
        for cache in self.bucket_caches.values():
            cache.clear()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Route samples to appropriate bucket caches."""
        # Extract frames and metadata
        if isinstance(batch, Stage1Batch):
            frames = _stage1_batch_to_frames(batch)
            metadata_list = _stage1_batch_to_metadata(batch)
        elif isinstance(batch, dict):
            frames = batch["frames"]
            metadata_list = self._extract_metadata(batch)
        else:
            frames = batch
            metadata_list = [{} for _ in range(frames.shape[0])]

        # Compute codes if any strategy needs them
        codes = None
        latents = None
        if self._any_strategy_needs_codes():
            with torch.no_grad():
                device = pl_module.device
                frames_gpu = frames.to(device)
                codes = pl_module.model(frames_gpu, return_only_codebook_ids=True).cpu()
                latents = pl_module.model.vq.codebooks[codes.to(device)].cpu()

        # Route each sample to matching bucket(s) AND global cache
        # Note: add_sample() internally handles is_full() for frame storage,
        # but always captures codes to history_codes for histogram strategies
        for i, meta in enumerate(metadata_list):
            frame = frames[i:i+1].detach().cpu()
            code = codes[i:i+1] if codes is not None else None
            latent = latents[i:i+1] if latents is not None else None

            # Add to global cache
            self.global_cache.add_sample(frame, meta, code, latent)

            # Route to matching buckets
            for bucket_name, bucket_cfg in self.bucket_configs.items():
                cache = self.bucket_caches[bucket_name]
                if bucket_cfg.matches(meta):
                    cache.add_sample(frame, meta, code, latent)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run strategies on their assigned buckets."""
        self.validation_count += 1

        # Update strategy counters
        for strategy in self.strategies:
            strategy.increment_count()

        # Select fixed samples for global cache on first validation
        if not self._first_full_validation_done:
            self._select_diverse_fixed_samples(self.global_cache, self.num_fixed_samples)
            for cache in self.bucket_caches.values():
                self._select_diverse_fixed_samples(cache, min(4, self.num_fixed_samples))
            self._first_full_validation_done = True

            # Log cache stats
            global_count = self.global_cache.sample_count()
            logger.info("✓ Global cache: %d samples", global_count)
            for name, cache in self.bucket_caches.items():
                count = cache.sample_count()
                holdout_tag = " (holdout)" if cache.is_holdout else ""
                logger.info("  [%s]%s: %d samples", name, holdout_tag, count)

        due = 0
        ran = 0
        skipped = 0
        soft_failed = 0

        # Run each strategy on its assigned buckets (read from strategy.buckets)
        for strategy in self.strategies:
            if not strategy.should_run():
                continue

            bucket_names = strategy.buckets  # Read directly from strategy

            # No bucket bindings -> use global cache
            if not bucket_names:
                due += 1
                can_run, reason = strategy.can_run(self.global_cache)
                if not can_run:
                    skipped += 1
                    logger.warning("[Stage1Validation] skip %s: %s", strategy.name, reason)
                    continue
                try:
                    result = strategy.run(self.global_cache, pl_module, trainer)
                    no_output, reason = self._result_is_no_output(result)
                    if no_output:
                        skipped += 1
                        logger.warning("[Stage1Validation] skip %s: %s", strategy.name, reason)
                    else:
                        ran += 1
                except Exception as e:
                    soft_failed += 1
                    logger.warning("[Stage1Validation] %s failed: %s", strategy.name, e)
                continue

            for bucket_name in bucket_names:
                due += 1
                if bucket_name not in self.bucket_caches:
                    skipped += 1
                    logger.warning("[Stage1Validation] bucket '%s' not found for %s", bucket_name, strategy.name)
                    continue
                
                cache = self.bucket_caches[bucket_name]
                can_run, reason = strategy.can_run(cache)
                
                if not can_run:
                    skipped += 1
                    logger.warning("[Stage1Validation] skip %s on %s: %s", strategy.name, bucket_name, reason)
                    continue

                # Always suffix bucket-bound metrics/images with bucket name to avoid
                # collisions between different per-bucket strategy instances.
                suffix = f"_{bucket_name}"
                if getattr(cache, "is_holdout", False):
                    suffix += "_holdout"
                
                try:
                    result = strategy.run(cache, pl_module, trainer, metric_suffix=suffix)
                    no_output, reason = self._result_is_no_output(result)
                    if no_output:
                        skipped += 1
                        logger.warning("[Stage1Validation] skip %s%s: %s", strategy.name, suffix, reason)
                    else:
                        ran += 1
                except Exception as e:
                    soft_failed += 1
                    logger.warning("[Stage1Validation] %s%s failed: %s", strategy.name, suffix, e)

        logger.info(
            "[Stage1Validation] due=%d ran=%d skipped=%d soft_failed=%d",
            due,
            ran,
            skipped,
            soft_failed,
        )

        # Run garbage collection after validation to keep loader + viz memory stable.
        if self.run_gc_after_validation:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class EMACallback(Callback):
    """
    Exponential Moving Average callback.

    Maintains EMA of model weights during training (LAPA style).

    Args:
        decay: EMA decay rate (default: 0.999)
        update_every: Update EMA every N steps (default: 1)
        update_after_step: Start EMA updates after N steps (default: 0)
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 1,
        update_after_step: int = 0,
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.ema_model = None
        self.num_updates = 0

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Initialize EMA model."""
        # Clone model for EMA
        # Convert OmegaConf to dict if needed
        model_config = dict(pl_module.model_config) if hasattr(pl_module.model_config, 'items') else pl_module.model_config
        self.ema_model = type(pl_module.model)(
            **model_config
        ).to(pl_module.device)
        self.ema_model.load_state_dict(pl_module.model.state_dict())
        self.ema_model.eval()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after training step."""
        if trainer.global_step < self.update_after_step:
            return

        if trainer.global_step % self.update_every != 0:
            return

        # Update EMA weights
        self.num_updates += 1
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                pl_module.model.parameters(),
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def state_dict(self):
        """Save EMA state."""
        return {
            "ema_model": self.ema_model.state_dict() if self.ema_model else None,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state."""
        if state_dict["ema_model"] is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dict["ema_model"])
        self.num_updates = state_dict["num_updates"]
