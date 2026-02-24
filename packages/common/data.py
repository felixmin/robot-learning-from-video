"""Data utilities for local OpenX indexed loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)



def oxe_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate OXE/OpenX samples by stacking frames and list-collecting metadata."""
    if not batch:
        return {}

    frames_list = []
    metadata: Dict[str, List[Any]] = {}
    for key in batch[0].keys():
        if key != "frames":
            metadata[key] = []

    for item in batch:
        frames_list.append(item["frames"])
        for key, value in item.items():
            if key != "frames" and key in metadata:
                metadata[key].append(value)

    result: Dict[str, Any] = {"frames": torch.stack(frames_list)}
    result.update(metadata)
    return result


STANDARD_BATCH_KEYS = frozenset(
    {
        "frames",
        "episode_id",
        "frame_idx",
        "dataset_name",
        "language",
    }
)


def validate_batch_keys(
    batch: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    raise_on_missing: bool = True,
) -> bool:
    """Validate required keys for LAQ training batches."""
    if required_keys is None:
        required_keys = list(STANDARD_BATCH_KEYS)

    missing_keys = [key for key in required_keys if key not in batch]
    if missing_keys:
        msg = f"Batch missing required keys: {missing_keys}. Present keys: {list(batch.keys())}"
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)
        return False

    extra_keys = set(batch.keys()) - STANDARD_BATCH_KEYS - {
        "metadata",
        "second_frame_idx",
        "environment",
        "task",
        "offset",
        "action",
        "initial_state",
        "dataset_type",
    }
    if extra_keys:
        logger.debug("Batch contains extra keys (not an error): %s", sorted(extra_keys))

    return True


class OpenXLocalDataModule(pl.LightningDataModule):
    """Local OpenX DataModule for indexed_full mode."""

    def __init__(
        self,
        *,
        datasets: List[Dict[str, Any]],
        preprocess: Dict[str, Any],
        loader: Dict[str, Any],
        adapter: Dict[str, Any],
    ):
        super().__init__()
        self.datasets = datasets
        self.preprocess = preprocess
        self.loader = loader
        self.adapter = adapter
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self._resolved_datasets: List[Dict[str, Any]] = []

    def setup(self, stage: Optional[str] = None):
        # Scripts may call setup() eagerly; Lightning will call setup() again.
        if self.train_dataset is not None and self.val_dataset is not None:
            logger.info("✓ OpenX local DataModule already initialized; skipping setup")
            return

        from common.adapters.openx_local import discover_local_subdatasets
        from common.adapters.openx_local_indexed_full import (
            OpenXLocalIndexedEpisodePairSampler,
            OpenXLocalIndexedPairMapDataset,
            prepare_openx_local_episode_index,
        )

        image_size = int(self.preprocess["image_size"])
        return_metadata = bool(self.preprocess["return_metadata"])

        local_cfg = self.adapter["openx_local"]
        root = str(local_cfg["root"])
        mode = str(local_cfg["mode"]).strip().lower()
        max_shards_per_dataset = local_cfg["max_shards_per_dataset"]
        pairs_per_episode = local_cfg["pairs_per_episode"]
        index_workers = int(local_cfg["index_workers"])
        index_cache_dir = local_cfg["index_cache_dir"]
        index_rebuild = bool(local_cfg["index_rebuild"])
        index_max_open_shards = int(local_cfg["index_max_open_shards"])
        weights_by_size = bool(local_cfg["weights_by_size"])
        auto_discover = bool(local_cfg.get("auto_discover", False))
        auto_train_split = str(local_cfg.get("auto_train_split", "train[:90%]"))
        auto_val_split = str(local_cfg.get("auto_val_split", "train[90%:]"))
        auto_pair_offset_steps = int(local_cfg.get("auto_pair_offset_steps", 5))
        auto_weight = float(local_cfg.get("auto_weight", 1.0))
        seed = int(local_cfg["seed"])
        resample_each_epoch = bool(local_cfg["resample_each_epoch"])
        stopping_strategy = str(local_cfg.get("stopping_strategy", "all_exhausted"))
        steps_per_epoch = local_cfg.get("steps_per_epoch")
        if steps_per_epoch is not None and int(steps_per_epoch) <= 0:
            raise ValueError(
                "data.adapter.openx_local.steps_per_epoch must be > 0 when set"
            )
        train_num_samples = (
            int(steps_per_epoch) * int(self.loader["batch_size"])
            if steps_per_epoch is not None
            else None
        )

        max_shards_opt = (
            int(max_shards_per_dataset)
            if max_shards_per_dataset is not None and int(max_shards_per_dataset) > 0
            else None
        )
        pairs_per_episode_opt = (
            int(pairs_per_episode)
            if pairs_per_episode is not None and int(pairs_per_episode) > 0
            else None
        )

        if auto_discover:
            discovered = discover_local_subdatasets(root)
            if not discovered:
                raise ValueError(f"No local OpenX datasets discovered under: {root}")
            resolved_datasets = [
                {
                    "name": dataset_name,
                    "train_split": auto_train_split,
                    "val_split": auto_val_split,
                    "pair_offset_steps": auto_pair_offset_steps,
                    "weight": auto_weight,
                    "approx_num_pairs": None,
                }
                for dataset_name in discovered
            ]
            logger.info(
                "✓ OpenX local auto-discovered %d datasets from %s",
                len(resolved_datasets),
                root,
            )
        else:
            resolved_datasets = list(self.datasets)

        if not resolved_datasets:
            raise ValueError("OpenX local has no datasets configured or discovered")
        self._resolved_datasets = resolved_datasets

        if mode != "indexed_full":
            raise ValueError(
                f"data.adapter.openx_local.mode must be 'indexed_full', got {mode!r}"
            )

        if index_cache_dir is None or str(index_cache_dir).strip() == "":
            raise ValueError(
                "data.adapter.openx_local.index_cache_dir must be set for mode='indexed_full'"
            )
        index_dir = str(Path(str(index_cache_dir)).expanduser().resolve() / "episode_index")

        train_index = prepare_openx_local_episode_index(
            root=root,
            dataset_entries=list(self._resolved_datasets),
            split_key="train_split",
            max_shards_per_dataset=max_shards_opt,
            index_workers=index_workers,
            index_dir=index_dir,
            rebuild=index_rebuild,
        )
        val_index = prepare_openx_local_episode_index(
            root=root,
            dataset_entries=list(self._resolved_datasets),
            split_key="val_split",
            max_shards_per_dataset=max_shards_opt,
            index_workers=index_workers,
            index_dir=index_dir,
            rebuild=index_rebuild,
        )

        self.train_dataset = OpenXLocalIndexedPairMapDataset(
            index=train_index,
            image_size=image_size,
            return_metadata=return_metadata,
            max_open_shards=index_max_open_shards,
        )
        self.val_dataset = OpenXLocalIndexedPairMapDataset(
            index=val_index,
            image_size=image_size,
            return_metadata=return_metadata,
            max_open_shards=index_max_open_shards,
        )
        self.train_sampler = OpenXLocalIndexedEpisodePairSampler(
            index=train_index,
            pairs_per_episode=pairs_per_episode_opt,
            weights_by_size=weights_by_size,
            num_samples=train_num_samples,
            seed=seed,
            epoch=0,
            resample_each_epoch=resample_each_epoch,
            stopping_strategy=stopping_strategy,
        )
        self.val_sampler = OpenXLocalIndexedEpisodePairSampler(
            index=val_index,
            pairs_per_episode=pairs_per_episode_opt,
            weights_by_size=False,
            num_samples=None,
            seed=seed + 1,
            epoch=0,
            resample_each_epoch=False,
            stopping_strategy=stopping_strategy,
        )

        dataset_names = [d["name"] for d in self._resolved_datasets]
        logger.info(
            "✓ OpenX local DataModule initialized (%s mode): %s",
            mode,
            ", ".join(dataset_names),
        )
        if steps_per_epoch is not None:
            logger.info(
                "✓ OpenX local train epoch size fixed to %d steps (%d samples)",
                int(steps_per_epoch),
                int(train_num_samples),
            )

    def train_dataloader(self):
        collate_fn = oxe_collate_fn if bool(self.preprocess["return_metadata"]) else None
        num_workers = int(self.loader["num_workers"])
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.loader["batch_size"]),
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.loader["pin_memory"]),
            prefetch_factor=self.loader["prefetch_factor"] if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        collate_fn = oxe_collate_fn if bool(self.preprocess["return_metadata"]) else None
        num_workers = int(self.loader["num_workers"])
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.loader["batch_size"]),
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.loader["pin_memory"]),
            prefetch_factor=self.loader["prefetch_factor"] if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self._resolved_datasets = []
