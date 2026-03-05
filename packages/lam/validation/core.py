"""
Core infrastructure for LAM validation.

Contains base classes and data structures:
- ValidationCache: Stores frames/latents
- ValidationStrategy: Base class for all strategies
- BucketConfig: Defines data subsets
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import lightning.pytorch as pl

from common.filters import matches_filters

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Essential metadata keys to cache (RAM safety - prevents Bridge metadata bloat)
# Only these keys are retained when caching samples to buckets
# Uses standardized keys from the unified batch interface
ESSENTIAL_METADATA_KEYS = frozenset({
    "dataset_name",    # Primary source identifier (e.g., "youtube", "bridge", "language_table")
    "dataset_short",   # Source alias derived from repo_id
    "action",          # For action scatter strategies (only first 2 dims used)
    "initial_state",   # For state scatter strategies (only first 2 dims used)
    "language",        # Task descriptions/instructions
    "scene_id",        # Scene identifier
    "environment",     # Environment identifier
    "task",            # Task identifier
})


def prune_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prune metadata to essential keys only (RAM safety)."""
    return {k: v for k, v in metadata.items() if k in ESSENTIAL_METADATA_KEYS}


@dataclass
class BucketConfig:
    """Configuration for a validation data bucket."""
    name: str
    filters: Dict[str, Any] = field(default_factory=dict)
    max_samples: int = 100
    is_holdout: bool = False  # True if this is OOD/distribution shift data

    def matches(self, metadata: Dict[str, Any]) -> bool:
        """Check if sample metadata matches this bucket's filters."""
        return matches_filters(metadata, self.filters)


@dataclass
class ValidationCache:
    """Cache for validation data across batches.

    Dual storage for codes:
    - codes: Bounded storage matching frames (for strategies needing frame correspondence)
    - all_codes: Unbounded storage for ALL validation codes (lightweight, ~100KB for 3k samples)

    This allows histogram strategies to see the true distribution across all validation
    samples, while keeping frame storage bounded for memory safety.
    """
    frames: List[torch.Tensor] = field(default_factory=list)
    latents: List[torch.Tensor] = field(default_factory=list)
    codes: List[torch.Tensor] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    # Unbounded code storage for histogram strategies (lightweight - just indices)
    all_codes: List[torch.Tensor] = field(default_factory=list)

    # Metadata for each sample (dataset_name, scene_id, etc.)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    # Fixed samples for consistent visualization (set once, reused)
    fixed_frames: Optional[torch.Tensor] = None
    fixed_indices: Optional[List[int]] = None
    fixed_metadata: Optional[List[Dict[str, Any]]] = None

    # Training samples (cached separately)
    train_frames: Optional[torch.Tensor] = None
    train_metadata: Optional[List[Dict[str, Any]]] = None

    # Bucket info (set when this cache belongs to a bucket)
    bucket_name: Optional[str] = None
    is_holdout: bool = False

    # Sample count tracking
    _sample_count: int = 0
    max_samples: int = 256

    def clear(self):
        """Clear all cached data (but keep fixed samples and train samples)."""
        self.frames.clear()
        self.latents.clear()
        self.codes.clear()
        self.losses.clear()
        self.metadata.clear()
        self.all_codes.clear()
        self._sample_count = 0

    def is_full(self) -> bool:
        """Check if cache has reached max_samples."""
        return self._sample_count >= self.max_samples

    def add_sample(
        self,
        frame: torch.Tensor,
        meta: Dict[str, Any],
        code: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        prune: bool = True,
    ):
        """Add a single sample to the cache.

        Args:
            frame: Frame tensor to cache
            meta: Metadata dictionary
            code: Optional codebook indices
            latent: Optional latent representation
            prune: If True, prune metadata to essential keys only (RAM safety)
        """
        # Always capture codes to all_codes (unbounded) for histogram strategies
        if code is not None:
            self.all_codes.append(code.cpu() if code.is_cuda else code)

        # Frame storage is bounded by max_samples
        if self.is_full():
            return

        self.frames.append(frame.cpu() if frame.is_cuda else frame)
        cached_meta = prune_metadata(meta) if prune else meta
        self.metadata.append([cached_meta])  # Always store as list for consistency with add_batch
        self._sample_count += 1

        if code is not None:
            self.codes.append(code.cpu() if code.is_cuda else code)
        if latent is not None:
            self.latents.append(latent.cpu() if latent.is_cuda else latent)

    def add_batch(
        self,
        frames: torch.Tensor,
        metadata_list: List[Dict[str, Any]],
        codes: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        prune: bool = True,
    ):
        """Add a batch of samples to the cache, respecting max_samples.

        Args:
            frames: Batch of frame tensors
            metadata_list: List of metadata dictionaries
            codes: Optional batch of codebook indices
            latents: Optional batch of latent representations
            prune: If True, prune metadata to essential keys only (RAM safety)
        """
        # Always capture ALL codes to all_codes (unbounded) for histogram strategies
        if codes is not None:
            self.all_codes.append(codes.cpu() if codes.is_cuda else codes)

        # Frame storage is bounded by max_samples
        remaining = self.max_samples - self._sample_count
        if remaining <= 0:
            return

        n_to_add = min(frames.shape[0], remaining)

        self.frames.append(frames[:n_to_add].cpu() if frames.is_cuda else frames[:n_to_add])
        if prune:
            cached_metadata = [prune_metadata(m) for m in metadata_list[:n_to_add]]
        else:
            cached_metadata = metadata_list[:n_to_add]
        self.metadata.append(cached_metadata)
        self._sample_count += n_to_add

        if codes is not None:
            self.codes.append(codes[:n_to_add].cpu() if codes.is_cuda else codes[:n_to_add])
        if latents is not None:
            self.latents.append(latents[:n_to_add].cpu() if latents.is_cuda else latents[:n_to_add])

    def sample_count(self) -> int:
        """Return current number of samples in cache."""
        return self._sample_count

    def get_all_frames(self) -> Optional[torch.Tensor]:
        """Concatenate all cached frames."""
        if not self.frames:
            return None
        return torch.cat(self.frames, dim=0)

    def get_all_latents(self) -> Optional[torch.Tensor]:
        """Concatenate all cached latents."""
        if not self.latents:
            return None
        return torch.cat(self.latents, dim=0)

    def get_codes(self) -> Optional[torch.Tensor]:
        """Concatenate cached codes (bounded by max_samples, has frame correspondence)."""
        if not self.codes:
            return None
        return torch.cat(self.codes, dim=0)

    def get_all_codes(self) -> Optional[torch.Tensor]:
        """Concatenate all codes (unbounded, all validation samples).

        Use this for histogram strategies that need the true distribution
        across all validation samples, not just the bounded subset.
        """
        if not self.all_codes:
            return None
        return torch.cat(self.all_codes, dim=0)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Flatten all metadata lists."""
        result = []
        for meta_batch in self.metadata:
            if isinstance(meta_batch, list):
                result.extend(meta_batch)
            else:
                result.append(meta_batch)
        return result

    def get_frames_by_filter(
        self,
        filters: Dict[str, Any],
        frames: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        """
        Get frames and metadata matching filter criteria.

        Args:
            filters: Dict of {key: value} or {key: [values]} to match.
                     Supports operators like ["!=", value] or [">", value].
            frames: Optional frames to filter (uses cached if None)
            metadata: Optional metadata to filter (uses cached if None)

        Returns:
            Tuple of (filtered_frames, filtered_metadata)
        """
        if frames is None:
            frames = self.get_all_frames()
        if metadata is None:
            metadata = self.get_all_metadata()

        if frames is None or not metadata:
            return None, []

        # Find matching indices
        indices = []
        for i, meta in enumerate(metadata):
            if matches_filters(meta, filters):
                indices.append(i)

        if not indices:
            return None, []

        filtered_frames = frames[indices]
        filtered_metadata = [metadata[i] for i in indices]
        return filtered_frames, filtered_metadata

    def count_samples_with_metadata(self, required_keys: List[str]) -> int:
        """Count samples that have all required metadata keys with non-None values."""
        all_metadata = self.get_all_metadata()
        count = 0
        for meta in all_metadata:
            has_all = True
            for key in required_keys:
                val = meta.get(key)
                if val is None:
                    has_all = False
                    break
                # Check for 2D+ actions (need at least 2 dims for scatter plots)
                if key == "action" and isinstance(val, (list, tuple)) and len(val) < 2:
                    has_all = False
                    break
            if has_all:
                count += 1
        return count

    def get_frames_by_dataset_name(self, dataset_name: str) -> Optional[torch.Tensor]:
        """Get frames filtered by dataset name (convenience method)."""
        frames, _ = self.get_frames_by_filter({"dataset_name": dataset_name})
        return frames

    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get count of samples per dataset name."""
        all_metadata = self.get_all_metadata()
        distribution: Dict[str, int] = {}
        for meta in all_metadata:
            dname = meta.get("dataset_short", "unknown")
            distribution[dname] = distribution.get(dname, 0) + 1
        return distribution


class ValidationStrategy(ABC):
    """
    Base class for validation strategies.

    Each strategy decides:
    - When to run (via should_run)
    - What data it needs (via required_metadata, min_samples)
    - What to compute and log (via run)

    Strategies are self-contained with bucket bindings:
    - buckets: List of bucket names this strategy operates on
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        buckets: Optional[List[str]] = None,
        **kwargs,
    ):
        self.name = name
        self.enabled = enabled
        self.every_n_validations = every_n_validations
        self.min_samples = min_samples
        self.buckets = buckets or []  # Empty = use global cache
        self.validation_count = 0

    def should_run(self) -> bool:
        """Check if this strategy should run on current validation."""
        if not self.enabled:
            return False
        return (self.validation_count % self.every_n_validations) == 0

    def will_run_next(self) -> bool:
        """Check if this strategy will run after the next increment.

        Used during batch processing to predict if codes need to be computed
        before the validation_count is incremented at epoch end.
        """
        if not self.enabled:
            return False
        return ((self.validation_count + 1) % self.every_n_validations) == 0

    def increment_count(self):
        """Increment validation counter."""
        self.validation_count += 1

    def required_metadata(self) -> List[str]:
        """
        Return list of metadata keys this strategy requires.

        Override in subclasses to declare requirements.
        Empty list means strategy works with any data.
        """
        return []

    @staticmethod
    def no_output(reason: str) -> Dict[str, Any]:
        """Return structured no-output result for callback-level accounting."""
        return {"_produced": 0, "_reason": str(reason)}

    @staticmethod
    def success(*, produced: int = 1, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return structured success result for callback-level accounting."""
        out = dict(metrics or {})
        out["_produced"] = max(0, int(produced))
        if out["_produced"] > 0:
            out.pop("_reason", None)
        return out

    def can_run(self, cache: ValidationCache) -> Tuple[bool, str]:
        """
        Check if strategy has sufficient applicable data in cache.

        Returns:
            Tuple of (can_run, reason_if_not)
        """
        # Check minimum sample count
        sample_count = cache.sample_count()
        if sample_count < self.min_samples:
            return False, f"Only {sample_count} samples (need {self.min_samples})"

        # Check required metadata
        required = self.required_metadata()
        if required:
            count_with_meta = cache.count_samples_with_metadata(required)
            if count_with_meta < self.min_samples:
                return False, f"Only {count_with_meta} samples with {required} (need {self.min_samples})"

        return True, ""

    @abstractmethod
    def needs_caching(self) -> bool:
        """Return True if this strategy needs data cached during validation."""
        pass

    def needs_codes(self) -> bool:
        """Return True if this strategy needs codebook indices cached."""
        return False

    @abstractmethod
    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Run the validation strategy.

        Args:
            cache: Cached validation data
            pl_module: The Lightning module
            trainer: The trainer
            metric_suffix: Suffix for metric names (e.g., "_bridge_holdout" for bucket-specific logging)

        Returns:
            Dict of metrics to log
        """
        pass

    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Get WandB logger from trainer."""
        if not WANDB_AVAILABLE:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger
        return None
