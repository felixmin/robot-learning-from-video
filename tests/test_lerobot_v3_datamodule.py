from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from common.data_factory import create_datamodule
from common.lerobot_v3_data import LeRobotV3DataModule
from common.lerobot_v3_source import CompiledEpisodeIndex, CompiledSourceIndex
from common.lerobot_v3_types import DatasetSample, Stage1Batch


def _compiled_index(*, repo_id: str, start: int, stop: int) -> CompiledSourceIndex:
    count = max(0, int(stop - start))
    return CompiledSourceIndex(
        repo_id=repo_id,
        fps=10,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key="observation.state",
        action_key="action",
        episodes=CompiledEpisodeIndex(
            episode_index=np.asarray([0], dtype=np.int32),
            dataset_from_index=np.asarray([start], dtype=np.int64),
            dataset_to_index=np.asarray([stop], dtype=np.int64),
            valid_anchor_start=np.asarray([start], dtype=np.int64),
            valid_anchor_end=np.asarray([stop], dtype=np.int64),
            valid_anchor_count=np.asarray([count], dtype=np.int32),
        ),
        sampleable_episode_ids=(
            np.asarray([0], dtype=np.int32)
            if count > 0
            else np.asarray([], dtype=np.int32)
        ),
        sampleable_episode_weights=(
            np.asarray([count], dtype=np.float64)
            if count > 0
            else np.asarray([], dtype=np.float64)
        ),
    )


class _FakeSource:
    def __init__(
        self,
        *,
        repo_id,
        root,
        revision,
        weight,
        camera_map,
        state_key,
        action_key,
        video_backend,
        tolerance_s,
    ):
        del root, revision, state_key, action_key, video_backend, tolerance_s
        self.repo_id = repo_id
        self.weight = weight
        self.camera_map = dict(camera_map)
        self.meta = SimpleNamespace(
            total_episodes=10,
            stats={},
            repo_id=repo_id,
            features={
                "observation.images.rgb": {"dtype": "video"},
                "observation.state": {"dtype": "float32"},
                "action": {"dtype": "float32"},
            },
        )
        self.compile_calls = []
        self.compiled_train_index = None
        self.compiled_val_index = None

    def compile(self, request, *, train_episode_indices, val_episode_indices):
        self.compile_calls.append(
            (request, set(train_episode_indices), set(val_episode_indices))
        )
        self.compiled_train_index = _compiled_index(
            repo_id=self.repo_id, start=0, stop=8
        )
        self.compiled_val_index = _compiled_index(
            repo_id=self.repo_id, start=8, stop=10
        )

    def get_sample(self, anchor_abs_index):
        raise AssertionError(
            "get_sample should not be called in datamodule wiring tests"
        )


def _cfg(*, num_sources: int = 1):
    sources = []
    for idx in range(num_sources):
        sources.append(
            {
                "repo_id": f"test/source_{idx}",
                "weight": float(idx + 1),
                "camera_map": {"primary": "observation.images.rgb"},
                "state_key": "observation.state",
                "action_key": "action",
                "val_episode_count": 2,
            }
        )
    return {
        "backend": "lerobot_v3",
        "preprocess": {"image_size": 224, "return_metadata": True},
        "loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 1,
        },
        "request": {
            "image_requests": {"primary": {"deltas_steps": [0, 1]}},
            "include_task_text": True,
            "include_metadata": True,
            "pad_missing_future": True,
            "image_size": [224, 224],
            "image_dtype": "uint8",
        },
        "output_format": "raw",
        "dataset": {"lerobot": {"sources": sources}},
        "adapter": {
            "lerobot_v3": {"seed": 7, "steps_per_epoch": 3, "resample_each_epoch": True}
        },
    }


def test_lerobot_v3_datamodule_builds_single_source_dataset(monkeypatch) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    dm = LeRobotV3DataModule(
        sources=_cfg(num_sources=1)["dataset"]["lerobot"]["sources"],
        request=_cfg(num_sources=1)["request"],
        loader=_cfg(num_sources=1)["loader"],
        adapter=_cfg(num_sources=1)["adapter"]["lerobot_v3"],
        output_format="raw",
    )
    dm.setup()
    assert len(dm.sources) == 1
    assert len(dm.train_sampler.compiled_sources) == 1
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None


def test_lerobot_v3_datamodule_builds_weighted_mixed_dataset(monkeypatch) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    cfg = _cfg(num_sources=2)
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="raw",
    )
    dm.setup()
    assert len(dm.sources) == 2
    assert np.allclose(
        dm.train_sampler.source_weights, np.asarray([1.0 / 3.0, 2.0 / 3.0])
    )


def test_lerobot_v3_datamodule_builds_separate_train_and_val_indices(
    monkeypatch,
) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    cfg = _cfg(num_sources=1)
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="raw",
    )
    dm.setup()
    _request, train_set, val_set = dm.sources[0].compile_calls[0]
    assert train_set == set(range(0, 8))
    assert val_set == set(range(8, 10))


def test_create_datamodule_supports_lerobot_v3_backend(monkeypatch) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    dm = create_datamodule(_cfg(num_sources=1))
    assert isinstance(dm, LeRobotV3DataModule)


def test_lerobot_v3_datamodule_stage1_output_returns_stage1_batch(monkeypatch) -> None:
    class _SampleSource(_FakeSource):
        def get_sample(self, anchor_abs_index):
            del anchor_abs_index
            import torch

            return DatasetSample(
                image_streams={"primary": torch.ones((2, 3, 8, 8), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.ones((2,), dtype=torch.bool)},
                task_text="pick",
                meta={"dataset_name": self.repo_id, "frame_idx": 3},
            )

    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _SampleSource)
    cfg = _cfg(num_sources=1)
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="stage1",
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, Stage1Batch)
    assert tuple(batch.image_streams["primary"].shape) == (4, 2, 3, 8, 8)


def test_lerobot_v3_datamodule_rejects_removed_stage1_legacy_output(
    monkeypatch,
) -> None:
    class _SampleSource(_FakeSource):
        def get_sample(self, anchor_abs_index):
            del anchor_abs_index
            import torch

            return DatasetSample(
                image_streams={"primary": torch.ones((2, 3, 8, 8), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.ones((2,), dtype=torch.bool)},
                task_text="pick",
                meta={"dataset_name": self.repo_id, "frame_idx": 3},
            )

    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _SampleSource)
    cfg = _cfg(num_sources=1)
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="stage1_legacy",
    )
    dm.setup()
    with pytest.raises(ValueError, match="Unsupported LeRobot output_format"):
        next(iter(dm.train_dataloader()))


def test_lerobot_v3_datamodule_uses_distributed_sampler_when_initialized(
    monkeypatch,
) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    monkeypatch.setattr(
        "common.lerobot_v3_data.torch.distributed.is_available", lambda: True
    )
    monkeypatch.setattr(
        "common.lerobot_v3_data.torch.distributed.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "common.lerobot_v3_data.torch.distributed.get_world_size", lambda: 4
    )
    monkeypatch.setattr("common.lerobot_v3_data.torch.distributed.get_rank", lambda: 1)

    cfg = _cfg(num_sources=2)
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="raw",
    )
    dm.setup()

    assert (
        dm.train_sampler.__class__.__name__ == "DistributedWeightedLeRobotTokenSampler"
    )
    assert dm.val_sampler.__class__.__name__ == "DistributedWeightedLeRobotTokenSampler"
    assert dm.train_sampler.world_size == 4
    assert dm.train_sampler.rank == 1
    assert dm.train_sampler.global_num_samples == 48


def test_lerobot_v3_datamodule_fails_fast_on_missing_camera_role_mapping(
    monkeypatch,
) -> None:
    monkeypatch.setattr("common.lerobot_v3_data.LeRobotSingleSource", _FakeSource)
    cfg = _cfg(num_sources=1)
    cfg["request"]["image_requests"] = {
        "primary": {"deltas_steps": [0, 1], "required": True},
        "wrist": {"deltas_steps": [0, 1], "required": False},
    }
    dm = LeRobotV3DataModule(
        sources=cfg["dataset"]["lerobot"]["sources"],
        request=cfg["request"],
        loader=cfg["loader"],
        adapter=cfg["adapter"]["lerobot_v3"],
        output_format="raw",
    )
    with pytest.raises(ValueError, match="missing camera role"):
        dm.setup()
