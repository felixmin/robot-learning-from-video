from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed
from torch.utils.data import DataLoader, Dataset

from common.lerobot_v3_adapters import (
    dataset_batch_to_stage2_batch,
    dataset_batch_to_stage1_batch,
)
from common.lerobot_v3_sampler import DistributedWeightedLeRobotTokenSampler
from common.lerobot_v3_sampler import WeightedLeRobotTokenSampler
from common.lerobot_v3_source import LeRobotSingleSource
from common.lerobot_v3_stats import build_run_normalization_stats
from common.lerobot_v3_types import BatchedDatasetSample, DatasetSample
from common.lerobot_v3_types import DatasetRequest, TemporalFieldRequest

try:
    import lightning.pytorch as pl
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised only in lightweight test envs

    class _LightningDataModule:
        pass

    class _PLNamespace:
        LightningDataModule = _LightningDataModule

    pl = _PLNamespace()


def _stack_optional_tensors(
    values: list[torch.Tensor | None], *, field_name: str
) -> torch.Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(
            f"Inconsistent optional tensor field {field_name!r} across batch"
        )
    first_shape = tuple(present[0].shape)
    if all(tuple(value.shape) == first_shape for value in present):
        return torch.stack(present)

    ranks = {value.ndim for value in present}
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent tensor ranks for field {field_name!r}")

    prefix_shape = first_shape[:-1]
    if not all(tuple(value.shape[:-1]) == prefix_shape for value in present):
        raise ValueError(
            f"Inconsistent tensor shapes for field {field_name!r}: {[tuple(value.shape) for value in present]}"
        )

    target_dim = max(int(value.shape[-1]) for value in present)
    padded: list[torch.Tensor] = []
    for value in present:
        pad_width = int(target_dim - int(value.shape[-1]))
        if pad_width > 0:
            value = F.pad(value, (0, pad_width))
        padded.append(value)
    return torch.stack(padded)


def _collect_optional_strings(
    values: list[str | None], *, field_name: str
) -> list[str] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(
            f"Inconsistent optional string field {field_name!r} across batch"
        )
    return [str(value) for value in present]


def _collate_meta(meta_values: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    present = [value for value in meta_values if value is not None]
    if not present:
        return None
    if len(present) != len(meta_values):
        raise ValueError("Inconsistent meta presence across batch")
    keys = set(present[0].keys())
    for value in present[1:]:
        if set(value.keys()) != keys:
            raise ValueError("Inconsistent meta keys across batch")
    return {key: [value[key] for value in present] for key in keys}


def collate_dataset_samples(batch: list[DatasetSample]) -> BatchedDatasetSample:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    first = batch[0]
    image_streams = None
    image_padding_masks = None
    if first.image_streams is not None:
        keys = set(first.image_streams.keys())
        for item in batch[1:]:
            if item.image_streams is None or set(item.image_streams.keys()) != keys:
                raise ValueError("All samples must share the same camera keys")
        image_streams = {
            key: torch.stack([item.image_streams[key] for item in batch])
            for key in first.image_streams
        }
        if first.image_padding_masks is not None:
            for item in batch[1:]:
                if (
                    item.image_padding_masks is None
                    or set(item.image_padding_masks.keys()) != keys
                ):
                    raise ValueError(
                        "All samples must share the same image padding mask keys"
                    )
            image_padding_masks = {
                key: torch.stack([item.image_padding_masks[key] for item in batch])
                for key in first.image_padding_masks
            }

    return BatchedDatasetSample(
        image_streams=image_streams,
        image_padding_masks=image_padding_masks,
        state=_stack_optional_tensors(
            [item.state for item in batch], field_name="state"
        ),
        state_is_pad=_stack_optional_tensors(
            [item.state_is_pad for item in batch], field_name="state_is_pad"
        ),
        action=_stack_optional_tensors(
            [item.action for item in batch], field_name="action"
        ),
        action_is_pad=_stack_optional_tensors(
            [item.action_is_pad for item in batch], field_name="action_is_pad"
        ),
        task_text=_collect_optional_strings(
            [item.task_text for item in batch], field_name="task_text"
        ),
        subtask_text=_collect_optional_strings(
            [item.subtask_text for item in batch], field_name="subtask_text"
        ),
        meta=_collate_meta([item.meta for item in batch]),
    )


def _request_from_config(request_cfg: Mapping[str, Any]) -> DatasetRequest:
    image_cfg = request_cfg["image_requests"]
    image_requests = {
        str(role): TemporalFieldRequest(
            deltas_steps=tuple(int(x) for x in cfg["deltas_steps"]),
            required=bool(cfg.get("required", True)),
        )
        for role, cfg in image_cfg.items()
    }
    state_cfg = request_cfg.get("state_request")
    action_cfg = request_cfg.get("action_request")
    return DatasetRequest(
        image_requests=image_requests,
        state_request=(
            None
            if state_cfg is None
            else TemporalFieldRequest(
                deltas_steps=tuple(int(x) for x in state_cfg["deltas_steps"]),
                required=bool(state_cfg.get("required", True)),
            )
        ),
        action_request=(
            None
            if action_cfg is None
            else TemporalFieldRequest(
                deltas_steps=tuple(int(x) for x in action_cfg["deltas_steps"]),
                required=bool(action_cfg.get("required", True)),
            )
        ),
        include_task_text=bool(request_cfg.get("include_task_text", False)),
        include_subtask_text=bool(request_cfg.get("include_subtask_text", False)),
        include_metadata=bool(request_cfg.get("include_metadata", True)),
        pad_missing_future=bool(request_cfg.get("pad_missing_future", True)),
        image_size=(
            None
            if request_cfg.get("image_size") is None
            else tuple(int(x) for x in request_cfg["image_size"])
        ),
        image_dtype=str(request_cfg.get("image_dtype", "uint8")),
    )


def _resolve_split_episode_sets(
    total_episodes: int,
    *,
    train_episode_indices: Sequence[int] | None,
    val_episode_indices: Sequence[int] | None,
    val_episode_count: int | None,
) -> tuple[set[int], set[int]]:
    if train_episode_indices is not None or val_episode_indices is not None:
        if train_episode_indices is None or val_episode_indices is None:
            raise ValueError(
                "train_episode_indices and val_episode_indices must be provided together"
            )
        return {int(x) for x in train_episode_indices}, {
            int(x) for x in val_episode_indices
        }
    if val_episode_count is None:
        raise ValueError(
            "Each LeRobot source must define val_episode_count or explicit episode index splits"
        )
    count = int(val_episode_count)
    if count <= 0 or count >= int(total_episodes):
        raise ValueError("val_episode_count must be > 0 and < total_episodes")
    val_set = set(range(total_episodes - count, total_episodes))
    train_set = set(range(0, total_episodes - count))
    return train_set, val_set


def _validate_source_image_requests(
    source: LeRobotSingleSource,
    request: DatasetRequest,
) -> None:
    features = source.meta.features
    for role in request.image_requests:
        if role not in source.camera_map:
            raise ValueError(
                f"Source {source.repo_id!r} is missing camera role {role!r}. "
                "All requested camera roles must be mapped by every source."
            )
        dataset_key = source.camera_map[role]
        if dataset_key not in features:
            raise ValueError(
                f"Source {source.repo_id!r} camera key {dataset_key!r} for role {role!r} "
                "is not present in dataset features."
            )
        dtype = str(features[dataset_key].get("dtype"))
        if dtype not in {"image", "video"}:
            raise ValueError(
                f"Source {source.repo_id!r} camera key {dataset_key!r} for role {role!r} "
                f"must be an image/video feature, got dtype={dtype!r}."
            )


class LeRobotMixedMapDataset(Dataset):
    def __init__(self, *, sources: Sequence[LeRobotSingleSource], split: str) -> None:
        self.sources = list(sources)
        self.split = str(split)

    def __getitem__(self, token):
        source = self.sources[int(token.source_id)]
        return source.get_sample(int(token.anchor_abs_index))

    def __len__(self) -> int:
        total = 0
        for source in self.sources:
            compiled = (
                source.compiled_train_index
                if self.split == "train"
                else source.compiled_val_index
            )
            if compiled is None:
                continue
            total += int(compiled.episodes.valid_anchor_count.sum())
        return total


class LeRobotV3DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        sources: Sequence[Mapping[str, Any]],
        request: Mapping[str, Any],
        loader: Mapping[str, Any],
        adapter: Mapping[str, Any],
        output_format: str,
    ) -> None:
        super().__init__()
        self.sources_cfg = [dict(src) for src in sources]
        self.request_cfg = dict(request)
        self.loader_cfg = dict(loader)
        self.adapter_cfg = dict(adapter)
        self.output_format = str(output_format)
        self.sources: list[LeRobotSingleSource] = []
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.request: DatasetRequest | None = None
        self.normalization_stats = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if is_distributed else 0
        self.request = _request_from_config(self.request_cfg)
        self.sources = []
        for source_cfg in self.sources_cfg:
            source = LeRobotSingleSource(
                repo_id=str(source_cfg["repo_id"]),
                root=source_cfg.get("root"),
                revision=source_cfg.get("revision"),
                weight=float(source_cfg["weight"]),
                camera_map=dict(source_cfg["camera_map"]),
                state_key=source_cfg.get("state_key"),
                action_key=source_cfg.get("action_key"),
                video_backend=source_cfg.get("video_backend"),
                tolerance_s=source_cfg.get("tolerance_s"),
            )
            _validate_source_image_requests(source, self.request)
            train_set, val_set = _resolve_split_episode_sets(
                int(source.meta.total_episodes),
                train_episode_indices=source_cfg.get("train_episode_indices"),
                val_episode_indices=source_cfg.get("val_episode_indices"),
                val_episode_count=source_cfg.get("val_episode_count"),
            )
            source.compile(
                self.request,
                train_episode_indices=train_set,
                val_episode_indices=val_set,
            )
            self.sources.append(source)

        if rank == 0:
            for source in self.sources:
                source.prepare(lock=True)
        if is_distributed:
            torch.distributed.barrier()
            if rank != 0:
                for source in self.sources:
                    source.prepare()

        self.normalization_stats = build_run_normalization_stats(
            self.sources,
            weights_mode=str(self.adapter_cfg.get("weights_mode", "explicit")),
        )

        train_compiled = [source.compiled_train_index for source in self.sources]
        val_compiled = [source.compiled_val_index for source in self.sources]
        train_weights = np.asarray(
            [source.weight for source in self.sources], dtype=np.float64
        )
        val_weights = np.asarray(
            [source.weight for source in self.sources], dtype=np.float64
        )

        batch_size = int(self.loader_cfg["batch_size"])
        steps_per_epoch = self.adapter_cfg.get("steps_per_epoch")
        world_size = torch.distributed.get_world_size() if is_distributed else 1
        if steps_per_epoch is None:
            train_num_samples = int(
                sum(index.episodes.valid_anchor_count.sum() for index in train_compiled)
            )
        else:
            train_num_samples = int(steps_per_epoch) * batch_size
        val_num_samples = int(
            sum(index.episodes.valid_anchor_count.sum() for index in val_compiled)
        )
        seed = int(self.adapter_cfg["seed"])

        self.train_dataset = LeRobotMixedMapDataset(sources=self.sources, split="train")
        self.val_dataset = LeRobotMixedMapDataset(sources=self.sources, split="val")
        if is_distributed:
            train_global_num_samples = train_num_samples * world_size
            val_global_num_samples = (
                (val_num_samples + world_size - 1) // world_size
            ) * world_size
            self.train_sampler = DistributedWeightedLeRobotTokenSampler(
                compiled_sources=train_compiled,
                source_weights=train_weights,
                global_num_samples=train_global_num_samples,
                rank=rank,
                world_size=world_size,
                seed=seed,
                epoch=0,
                resample_each_epoch=bool(
                    self.adapter_cfg.get("resample_each_epoch", True)
                ),
            )
            self.val_sampler = DistributedWeightedLeRobotTokenSampler(
                compiled_sources=val_compiled,
                source_weights=val_weights,
                global_num_samples=val_global_num_samples,
                rank=rank,
                world_size=world_size,
                seed=seed + 1,
                epoch=0,
                resample_each_epoch=False,
            )
        else:
            self.train_sampler = WeightedLeRobotTokenSampler(
                compiled_sources=train_compiled,
                source_weights=train_weights,
                num_samples=train_num_samples,
                seed=seed,
                epoch=0,
                resample_each_epoch=bool(
                    self.adapter_cfg.get("resample_each_epoch", True)
                ),
            )
            self.val_sampler = WeightedLeRobotTokenSampler(
                compiled_sources=val_compiled,
                source_weights=val_weights,
                num_samples=val_num_samples,
                seed=seed + 1,
                epoch=0,
                resample_each_epoch=False,
            )

    def _adapt_batch(self, batch: BatchedDatasetSample):
        if self.output_format == "raw":
            return batch
        if self.output_format == "stage1":
            return dataset_batch_to_stage1_batch(batch)
        if self.output_format == "stage2":
            return dataset_batch_to_stage2_batch(batch)
        raise ValueError(f"Unsupported LeRobot output_format {self.output_format!r}")

    def _collate_and_adapt(self, batch: list[DatasetSample]):
        return self._adapt_batch(collate_dataset_samples(batch))

    def train_dataloader(self):
        num_workers = int(self.loader_cfg["num_workers"])
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.loader_cfg["batch_size"]),
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.loader_cfg["pin_memory"]),
            prefetch_factor=(
                self.loader_cfg["prefetch_factor"] if num_workers > 0 else None
            ),
            persistent_workers=num_workers > 0,
            collate_fn=self._collate_and_adapt,
        )

    def val_dataloader(self):
        num_workers = int(self.loader_cfg["num_workers"])
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.loader_cfg["batch_size"]),
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(self.loader_cfg["pin_memory"]),
            prefetch_factor=(
                self.loader_cfg["prefetch_factor"] if num_workers > 0 else None
            ),
            persistent_workers=num_workers > 0,
            collate_fn=self._collate_and_adapt,
        )
