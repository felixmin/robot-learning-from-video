from __future__ import annotations

from dataclasses import dataclass
import fcntl
from pathlib import Path
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common.action_frame_filtering import build_action_frame_filter
from common.action_frame_filtering import normalize_filtering_config
from common.lerobot_v3_types import DatasetRequest
from common.lerobot_v3_types import DatasetSample

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


def _dataset_short_from_repo_id(repo_id: str) -> str:
    short = str(repo_id).split("/")[-1]
    if short.endswith("_lerobot"):
        short = short[: -len("_lerobot")]
    return short


@dataclass(frozen=True)
class CompiledEpisodeIndex:
    episode_index: np.ndarray
    dataset_from_index: np.ndarray
    dataset_to_index: np.ndarray
    valid_anchor_start: np.ndarray
    valid_anchor_end: np.ndarray
    valid_anchor_count: np.ndarray
    sample_anchor_offsets_start: np.ndarray | None = None
    sample_anchor_offsets_end: np.ndarray | None = None
    sample_anchor_values: np.ndarray | None = None


@dataclass(frozen=True)
class CompiledSourceIndex:
    repo_id: str
    fps: int
    camera_role_to_key: dict[str, str]
    state_key: str | None
    action_key: str | None
    episodes: CompiledEpisodeIndex
    sampleable_episode_ids: np.ndarray
    sampleable_episode_weights: np.ndarray
    filter_summary: dict[str, Any] | None = None
    filter_cache_path: str | None = None


@dataclass(frozen=True)
class SourceMetaView:
    repo_id: str
    fps: int
    episodes: list[dict[str, Any]]
    stats: dict[str, Any]


@dataclass
class LeRobotSourceRuntime:
    dataset: LeRobotDataset
    compiled_index: CompiledSourceIndex
    resolved_delta_timestamps: dict[str, list[float]]


def _steps_to_seconds(deltas_steps: tuple[int, ...], *, fps: int) -> list[float]:
    return [float(delta) / float(fps) for delta in deltas_steps]


def load_lerobot_meta(
    repo_id: str,
    root: str | None,
    revision: str | None,
) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)


def resolve_request_to_delta_timestamps(
    request: DatasetRequest,
    *,
    fps: int,
    camera_role_to_key: dict[str, str],
    state_key: str | None,
    action_key: str | None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for role, field_request in request.image_requests.items():
        if role not in camera_role_to_key:
            if field_request.required:
                raise KeyError(f"Missing camera role mapping for {role!r}")
            continue
        out[camera_role_to_key[role]] = _steps_to_seconds(
            field_request.deltas_steps, fps=fps
        )

    if request.state_request is not None and state_key is not None:
        out[state_key] = _steps_to_seconds(request.state_request.deltas_steps, fps=fps)
    if request.action_request is not None and action_key is not None:
        out[action_key] = _steps_to_seconds(
            request.action_request.deltas_steps, fps=fps
        )
    return out


def _episode_delta_bounds(
    request: DatasetRequest,
) -> tuple[int, int]:
    min_delta = 0
    max_delta = 0

    for field_request in request.image_requests.values():
        min_delta = min(min_delta, min(field_request.deltas_steps))
        max_delta = max(max_delta, max(field_request.deltas_steps))

    if request.state_request is not None:
        deltas = request.state_request.deltas_steps
        min_delta = min(min_delta, min(deltas))
        if not request.pad_missing_future or max(deltas) < 0:
            max_delta = max(max_delta, max(deltas))

    if request.action_request is not None:
        deltas = request.action_request.deltas_steps
        min_delta = min(min_delta, min(deltas))
        if not request.pad_missing_future or max(deltas) < 0:
            max_delta = max(max_delta, max(deltas))

    return int(min_delta), int(max_delta)


def compile_source_index(
    *,
    meta: SourceMetaView,
    request: DatasetRequest,
    camera_role_to_key: dict[str, str],
    state_key: str | None,
    action_key: str | None,
) -> CompiledSourceIndex:
    min_delta_steps, max_delta_steps = _episode_delta_bounds(request)

    episode_index: list[int] = []
    dataset_from_index: list[int] = []
    dataset_to_index: list[int] = []
    valid_anchor_start: list[int] = []
    valid_anchor_end: list[int] = []
    valid_anchor_count: list[int] = []
    sampleable_episode_ids: list[int] = []
    sampleable_episode_weights: list[float] = []

    for episode in meta.episodes:
        ep_idx = int(episode["episode_index"])
        ep_start = int(episode["dataset_from_index"])
        ep_end = int(episode["dataset_to_index"])

        start = ep_start - min_delta_steps
        end = ep_end - max_delta_steps
        count = max(0, end - start)

        episode_index.append(ep_idx)
        dataset_from_index.append(ep_start)
        dataset_to_index.append(ep_end)
        valid_anchor_start.append(start)
        valid_anchor_end.append(end)
        valid_anchor_count.append(count)

        if count > 0:
            sampleable_episode_ids.append(ep_idx)
            sampleable_episode_weights.append(float(count))

    episode_bundle = CompiledEpisodeIndex(
        episode_index=np.asarray(episode_index, dtype=np.int32),
        dataset_from_index=np.asarray(dataset_from_index, dtype=np.int64),
        dataset_to_index=np.asarray(dataset_to_index, dtype=np.int64),
        valid_anchor_start=np.asarray(valid_anchor_start, dtype=np.int64),
        valid_anchor_end=np.asarray(valid_anchor_end, dtype=np.int64),
        valid_anchor_count=np.asarray(valid_anchor_count, dtype=np.int32),
    )
    return CompiledSourceIndex(
        repo_id=str(meta.repo_id),
        fps=int(meta.fps),
        camera_role_to_key=dict(camera_role_to_key),
        state_key=state_key,
        action_key=action_key,
        episodes=episode_bundle,
        sampleable_episode_ids=np.asarray(sampleable_episode_ids, dtype=np.int32),
        sampleable_episode_weights=np.asarray(
            sampleable_episode_weights, dtype=np.float64
        ),
    )


def _filter_episodes(
    meta: LeRobotDatasetMetadata, episode_indices: set[int] | None
) -> SourceMetaView:
    if episode_indices is None:
        episodes = list(meta.episodes)
    else:
        episodes = [
            ep for ep in meta.episodes if int(ep["episode_index"]) in episode_indices
        ]

    # Some dataset shards can contain duplicate metadata rows for the same episode.
    # Keep the first occurrence to avoid overcounting sampleable episodes.
    deduped: list[dict[str, Any]] = []
    seen_episode_indices: set[int] = set()
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        if ep_idx in seen_episode_indices:
            continue
        seen_episode_indices.add(ep_idx)
        deduped.append(episode)

    duplicate_count = len(episodes) - len(deduped)
    if duplicate_count > 0:
        logger.warning(
            "Deduplicated %d duplicate episode metadata rows for repo=%s",
            duplicate_count,
            str(meta.repo_id),
        )

    return SourceMetaView(
        repo_id=str(meta.repo_id),
        fps=int(meta.fps),
        episodes=deduped,
        stats=dict(meta.stats),
    )


def _to_tchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor rank 3 or 4, got {tuple(image.shape)}")
    if int(image.shape[1]) == 3:
        return image
    if int(image.shape[-1]) == 3:
        return image.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unsupported image layout: {tuple(image.shape)}")


def _resize_image_stream_to_uint8(
    image: torch.Tensor, image_size: tuple[int, int] | None
) -> torch.Tensor:
    image_t = _to_tchw(image)
    if image_t.dtype == torch.uint8:
        image_f = image_t.to(torch.float32)
    elif torch.is_floating_point(image_t):
        image_f = image_t.to(torch.float32) * 255.0
    else:
        raise TypeError(f"Expected uint8 or floating image stream, got {image_t.dtype}")
    if image_size is not None and tuple(image_f.shape[-2:]) != tuple(image_size):
        image_f = F.interpolate(
            image_f, size=tuple(image_size), mode="bilinear", align_corners=False
        )
    return image_f.round().clamp_(0.0, 255.0).to(torch.uint8)


def _restrict_dataset_video_features(
    dataset: LeRobotDataset, requested_video_keys: set[str]
) -> None:
    if not requested_video_keys:
        return
    filtered_features = {
        key: value
        for key, value in dataset.meta.info["features"].items()
        if key not in dataset.meta.video_keys or key in requested_video_keys
    }
    dataset.meta.info["features"] = filtered_features


class LeRobotSingleSource:
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | None,
        revision: str | None,
        weight: float,
        camera_map: dict[str, str],
        state_key: str | None,
        action_key: str | None,
        video_backend: str | None = None,
        tolerance_s: float | None = None,
        filtering_cfg: dict[str, Any] | None = None,
        global_filtering_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.repo_id = str(repo_id)
        self.root = root
        self.revision = revision
        self.weight = float(weight)
        self.camera_map = dict(camera_map)
        self.state_key = state_key
        self.action_key = action_key
        self.video_backend = None if video_backend is None else str(video_backend)

        self.tolerance_s = tolerance_s
        self.filtering_cfg = None if filtering_cfg is None else dict(filtering_cfg)
        self.global_filtering_cfg = (
            None if global_filtering_cfg is None else dict(global_filtering_cfg)
        )
        self.meta = load_lerobot_meta(self.repo_id, root, revision)
        self.request: DatasetRequest | None = None
        self.compiled_train_index: CompiledSourceIndex | None = None
        self.compiled_val_index: CompiledSourceIndex | None = None
        self._runtime: LeRobotSourceRuntime | None = None

    def _resolved_root(self) -> Path:
        return (
            Path(self.root) if self.root is not None else HF_LEROBOT_HOME / self.repo_id
        )

    def _resolved_request(self) -> dict[str, list[float]]:
        if self.request is None:
            raise RuntimeError("Source must be compiled before use")
        return resolve_request_to_delta_timestamps(
            request=self.request,
            fps=int(self.meta.fps),
            camera_role_to_key=self.camera_map,
            state_key=self.state_key,
            action_key=self.action_key,
        )

    def compile(
        self,
        request: DatasetRequest,
        *,
        train_episode_indices: set[int] | None,
        val_episode_indices: set[int] | None,
    ) -> None:
        self.request = request
        train_meta = _filter_episodes(self.meta, train_episode_indices)
        val_meta = _filter_episodes(self.meta, val_episode_indices)
        self.compiled_train_index = compile_source_index(
            meta=train_meta,
            request=request,
            camera_role_to_key=self.camera_map,
            state_key=self.state_key,
            action_key=self.action_key,
        )
        self.compiled_val_index = compile_source_index(
            meta=val_meta,
            request=request,
            camera_role_to_key=self.camera_map,
            state_key=self.state_key,
            action_key=self.action_key,
        )

        logger.info(
            "LeRobot source compile repo=%s root=%s total_episodes=%d train_episodes=%d val_episodes=%d",
            self.repo_id,
            str(self.meta.root),
            int(self.meta.total_episodes),
            int(self.compiled_train_index.episodes.episode_index.shape[0]),
            int(self.compiled_val_index.episodes.episode_index.shape[0]),
        )

        normalized_filter_cfg = normalize_filtering_config(
            global_filtering=self.global_filtering_cfg,
            source_filtering=self.filtering_cfg,
        )
        if normalized_filter_cfg is not None and bool(
            normalized_filter_cfg.get("apply_at_sampling", True)
        ):
            self.compiled_train_index = self._apply_action_frame_filtering(
                compiled=self.compiled_train_index,
                split="train",
                filtering_cfg=normalized_filter_cfg,
            )
            self.compiled_val_index = self._apply_action_frame_filtering(
                compiled=self.compiled_val_index,
                split="val",
                filtering_cfg=normalized_filter_cfg,
            )

        self._runtime = None

    def _apply_action_frame_filtering(
        self,
        *,
        compiled: CompiledSourceIndex,
        split: str,
        filtering_cfg: dict[str, Any],
    ) -> CompiledSourceIndex:
        """Build/reuse action-frame filtering cache and project kept anchors into compiled index."""
        role = next(iter(self.request.image_requests.keys()))
        primary_camera_dataset_key = self.camera_map[role]
        motion_cfg = dict(filtering_cfg.get("motion", {}))
        use_all_cameras = bool(motion_cfg.get("aggregate_all_cameras", False))
        if use_all_cameras:
            camera_keys_from_map = [
                str(v) for v in self.camera_map.values() if v is not None
            ]
            camera_dataset_keys = tuple(dict.fromkeys(camera_keys_from_map))
            if len(camera_dataset_keys) == 0:
                camera_dataset_keys = tuple(
                    sorted(str(key) for key in self.meta.camera_keys)
                )
        else:
            camera_dataset_keys = (str(primary_camera_dataset_key),)
        camera_aggregate_reduce = str(motion_cfg.get("aggregate_reduce", "mean"))
        request_deltas = tuple(
            int(x) for x in self.request.image_requests[role].deltas_steps
        )

        logger.info(
            "Action-frame filtering source=%s split=%s start cameras=%s mode=%s candidate_episodes=%d",
            self.repo_id,
            split,
            list(camera_dataset_keys),
            str(filtering_cfg.get("mode")),
            int(compiled.episodes.episode_index.shape[0]),
        )

        result = build_action_frame_filter(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            video_backend=self.video_backend,
            tolerance_s=self.tolerance_s,
            request_image_deltas=request_deltas,
            camera_dataset_keys=camera_dataset_keys,
            camera_aggregate_reduce=camera_aggregate_reduce,
            action_key=self.action_key,
            episode_ids=compiled.episodes.episode_index,
            candidate_start=compiled.episodes.valid_anchor_start,
            candidate_end=compiled.episodes.valid_anchor_end,
            filtering_cfg=filtering_cfg,
            split=split,
        )

        counts = result.kept_counts.astype(np.int32)
        range_start = result.kept_range_start.astype(np.int64)
        range_end = result.kept_range_end.astype(np.int64)
        sampleable_episode_ids = compiled.episodes.episode_index[counts > 0].astype(
            np.int32
        )
        sampleable_episode_weights = counts[counts > 0].astype(np.float64)

        logger.info(
            "Action-frame filtering source=%s split=%s cache=%s before=%d after=%d trimmed=%.3f motion_only_removed=%d action_only_removed=%d",
            self.repo_id,
            split,
            result.summary.get("cache"),
            int(result.summary.get("anchors_before", 0)),
            int(result.summary.get("anchors_after", 0)),
            float(result.summary.get("trimmed_fraction", 0.0)),
            int(result.summary.get("motion_only_removed", 0)),
            int(result.summary.get("action_only_removed", 0)),
        )

        episode_bundle = CompiledEpisodeIndex(
            episode_index=compiled.episodes.episode_index,
            dataset_from_index=compiled.episodes.dataset_from_index,
            dataset_to_index=compiled.episodes.dataset_to_index,
            valid_anchor_start=range_start,
            valid_anchor_end=range_end,
            valid_anchor_count=counts,
            sample_anchor_offsets_start=result.kept_offsets_start.astype(np.int64),
            sample_anchor_offsets_end=result.kept_offsets_end.astype(np.int64),
            sample_anchor_values=result.kept_anchor_values.astype(np.int64),
        )

        return CompiledSourceIndex(
            repo_id=compiled.repo_id,
            fps=compiled.fps,
            camera_role_to_key=compiled.camera_role_to_key,
            state_key=compiled.state_key,
            action_key=compiled.action_key,
            episodes=episode_bundle,
            sampleable_episode_ids=sampleable_episode_ids,
            sampleable_episode_weights=sampleable_episode_weights,
            filter_summary=dict(result.summary),
            filter_cache_path=result.cache_path,
        )

    def prepare(self, *, lock: bool = False) -> None:
        """Create the LeRobotDataset and store as runtime.

        Must be called after compile() and before DataLoader workers fork.
        Use lock=True on rank 0 to safely trigger any dataset downloads.
        """
        if self._runtime is not None:
            return
        if self.request is None or self.compiled_train_index is None:
            raise RuntimeError("Source must be compiled before prepare()")
        resolved = self._resolved_request()
        requested_video_keys = {
            dataset_key
            for role, dataset_key in self.camera_map.items()
            if role in self.request.image_requests
        }
        root = self._resolved_root()
        if lock:
            root.mkdir(parents=True, exist_ok=True)
            lock_path = root / ".hlrp_materialize.lock"
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                dataset = self._create_dataset(resolved)
                _restrict_dataset_video_features(dataset, requested_video_keys)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            dataset = self._create_dataset(resolved)
            _restrict_dataset_video_features(dataset, requested_video_keys)
        self._runtime = LeRobotSourceRuntime(
            dataset=dataset,
            compiled_index=self.compiled_train_index,
            resolved_delta_timestamps=resolved,
        )

    def _create_dataset(
        self, delta_timestamps: dict[str, list[float]]
    ) -> LeRobotDataset:
        return LeRobotDataset(
            repo_id=self.repo_id,
            root=str(self._resolved_root()),
            revision=self.revision,
            delta_timestamps=delta_timestamps,
            video_backend=self.video_backend,
            tolerance_s=self.tolerance_s if self.tolerance_s is not None else 1e-4,
        )

    @property
    def runtime(self) -> LeRobotSourceRuntime:
        if self._runtime is None:
            raise RuntimeError("Source must be prepared before use (call prepare())")
        return self._runtime

    def get_sample(self, anchor_abs_index: int) -> DatasetSample:
        runtime = self.runtime
        if self.request is None:
            raise RuntimeError("Source request must be set before get_sample")
        if self.request.image_dtype != "uint8":
            raise NotImplementedError(self.request.image_dtype)
        raw = runtime.dataset[int(anchor_abs_index)]
        image_streams: dict[str, torch.Tensor] = {}
        image_padding_masks: dict[str, torch.Tensor] = {}
        for role, dataset_key in self.camera_map.items():
            if role not in self.request.image_requests:
                continue
            if dataset_key not in raw:
                continue
            image_streams[role] = _resize_image_stream_to_uint8(
                raw[dataset_key], self.request.image_size
            )
            is_pad_key = f"{dataset_key}_is_pad"
            if is_pad_key in raw:
                image_padding_masks[role] = ~torch.as_tensor(
                    raw[is_pad_key], dtype=torch.bool
                )
            else:
                image_padding_masks[role] = torch.ones(
                    (int(image_streams[role].shape[0]),), dtype=torch.bool
                )

        state = None
        state_is_pad = None
        if (
            self.request.state_request is not None
            and self.state_key is not None
            and self.state_key in raw
        ):
            state = torch.as_tensor(raw[self.state_key])
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state_is_pad = torch.as_tensor(
                raw.get(
                    f"{self.state_key}_is_pad",
                    torch.zeros((int(state.shape[0]),), dtype=torch.bool),
                )
            )

        action = None
        action_is_pad = None
        if (
            self.request.action_request is not None
            and self.action_key is not None
            and self.action_key in raw
        ):
            action = torch.as_tensor(raw[self.action_key])
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action_is_pad = torch.as_tensor(
                raw.get(
                    f"{self.action_key}_is_pad",
                    torch.zeros((int(action.shape[0]),), dtype=torch.bool),
                )
            )

        return DatasetSample(
            image_streams=image_streams or None,
            image_padding_masks=image_padding_masks or None,
            state=state,
            state_is_pad=state_is_pad,
            action=action,
            action_is_pad=action_is_pad,
            task_text=str(raw["task"]) if "task" in raw else None,
            subtask_text=str(raw["subtask"]) if "subtask" in raw else None,
            meta={
                "dataset_name": self.repo_id,
                "dataset_short": _dataset_short_from_repo_id(self.repo_id),
                "episode_id": (
                    int(raw["episode_index"]) if "episode_index" in raw else None
                ),
                "frame_idx": int(raw["frame_index"]) if "frame_index" in raw else None,
            },
        )
