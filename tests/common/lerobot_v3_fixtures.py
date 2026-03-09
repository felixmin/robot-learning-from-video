from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from common.lerobot_v3_types import DatasetRequest, TemporalFieldRequest


def make_test_request(
    *,
    image_requests: dict[str, tuple[int, ...]] | None = None,
    state_deltas: tuple[int, ...] | None = None,
    action_deltas: tuple[int, ...] | None = None,
    image_size: tuple[int, int] | None = (224, 224),
    pad_missing_future: bool = True,
) -> DatasetRequest:
    image_requests = image_requests or {"primary": (0, 5)}
    return DatasetRequest(
        image_requests={
            role: TemporalFieldRequest(deltas_steps=tuple(deltas))
            for role, deltas in image_requests.items()
        },
        state_request=(
            None
            if state_deltas is None
            else TemporalFieldRequest(deltas_steps=tuple(state_deltas))
        ),
        action_request=(
            None
            if action_deltas is None
            else TemporalFieldRequest(deltas_steps=tuple(action_deltas))
        ),
        include_metadata=True,
        pad_missing_future=pad_missing_future,
        image_size=image_size,
        image_dtype="uint8",
    )


def make_test_meta(
    *,
    repo_id: str = "test/repo",
    fps: int = 10,
    episodes: list[dict[str, Any]] | None = None,
    stats: dict[str, dict[str, np.ndarray]] | None = None,
):
    if episodes is None:
        episodes = [
            {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 10},
            {"episode_index": 1, "dataset_from_index": 10, "dataset_to_index": 20},
        ]
    return SimpleNamespace(
        repo_id=repo_id,
        fps=fps,
        episodes=episodes,
        stats=stats or {},
    )


def make_test_source_stats(
    *,
    action_mean: list[float],
    action_std: list[float],
    action_min: list[float] | None = None,
    action_max: list[float] | None = None,
    action_count: int = 100,
    state_mean: list[float] | None = None,
    state_std: list[float] | None = None,
    state_count: int = 100,
) -> dict[str, dict[str, np.ndarray]]:
    action_min = (
        action_min if action_min is not None else [m - 1.0 for m in action_mean]
    )
    action_max = (
        action_max if action_max is not None else [m + 1.0 for m in action_mean]
    )
    stats: dict[str, dict[str, np.ndarray]] = {
        "action": {
            "mean": np.asarray(action_mean, dtype=np.float64),
            "std": np.asarray(action_std, dtype=np.float64),
            "min": np.asarray(action_min, dtype=np.float64),
            "max": np.asarray(action_max, dtype=np.float64),
            "count": np.asarray([action_count], dtype=np.int64),
        }
    }
    if state_mean is not None and state_std is not None:
        stats["observation.state"] = {
            "mean": np.asarray(state_mean, dtype=np.float64),
            "std": np.asarray(state_std, dtype=np.float64),
            "min": np.asarray([m - 1.0 for m in state_mean], dtype=np.float64),
            "max": np.asarray([m + 1.0 for m in state_mean], dtype=np.float64),
            "count": np.asarray([state_count], dtype=np.int64),
        }
    return stats
