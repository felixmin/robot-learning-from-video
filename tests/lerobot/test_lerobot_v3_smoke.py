from __future__ import annotations

import os
from collections import Counter

import pytest
import torch

from common.lerobot_v3_data import LeRobotV3DataModule
from common.lerobot_v3_types import Stage1Batch
from stage2.backends.interfaces import Stage2Batch


def _source_cfg(
    *,
    repo_id: str,
    camera_map: dict[str, str],
    weight: float = 1.0,
    val_episode_count: int = 2,
) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "weight": weight,
        "camera_map": camera_map,
        "state_key": "observation.state",
        "action_key": "action",
        "val_episode_count": val_episode_count,
    }


def _request_cfg(
    *, image_requests: dict[str, list[int]], include_actions: bool = True
) -> dict[str, object]:
    out: dict[str, object] = {
        "image_requests": {
            role: {"deltas_steps": deltas} for role, deltas in image_requests.items()
        },
        "state_request": {"deltas_steps": [0]},
        "include_task_text": True,
        "include_metadata": True,
        "pad_missing_future": True,
        "image_size": [96, 96],
        "image_dtype": "uint8",
    }
    if include_actions:
        out["action_request"] = {"deltas_steps": [0, 1, 2]}
    return out


def _loader_cfg(batch_size: int = 4) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 1,
    }


def _adapter_cfg(*, steps_per_epoch: int = 4) -> dict[str, object]:
    return {
        "seed": 7,
        "steps_per_epoch": steps_per_epoch,
        "resample_each_epoch": True,
        "weights_mode": "explicit",
    }


pytestmark = pytest.mark.skipif(
    os.environ.get("HLRP_RUN_LEROBOT_SMOKE") != "1",
    reason="Set HLRP_RUN_LEROBOT_SMOKE=1 to enable live LeRobot smoke tests",
)


def test_real_single_source_stage1_and_stage2_batches() -> None:
    common_kwargs = dict(
        sources=[
            _source_cfg(
                repo_id="lerobot/nyu_rot_dataset",
                camera_map={"primary": "observation.images.image"},
                val_episode_count=2,
            )
        ],
        request=_request_cfg(image_requests={"primary": [0, 1]}),
        loader=_loader_cfg(batch_size=4),
        adapter=_adapter_cfg(steps_per_epoch=2),
    )

    stage1_dm = LeRobotV3DataModule(output_format="stage1", **common_kwargs)
    stage1_dm.setup()
    stage1_batch = next(iter(stage1_dm.train_dataloader()))
    assert isinstance(stage1_batch, Stage1Batch)
    assert tuple(stage1_batch.image_streams["primary"].shape[:3]) == (4, 2, 3)
    assert len(stage1_batch.task_text) == 4

    stage2_dm = LeRobotV3DataModule(output_format="stage2", **common_kwargs)
    stage2_dm.setup()
    stage2_batch = next(iter(stage2_dm.train_dataloader()))
    assert isinstance(stage2_batch, Stage2Batch)
    assert tuple(stage2_batch.image_streams["primary"].shape[:3]) == (4, 2, 3)
    assert tuple(stage2_batch.state.shape) == (4, 1, 7)
    assert tuple(stage2_batch.target_actions.shape) == (4, 3, 7)
    assert stage2_dm.normalization_stats is not None
    assert "action" in stage2_dm.normalization_stats


def test_real_mixed_source_yields_samples_from_both_sources() -> None:
    dm = LeRobotV3DataModule(
        sources=[
            _source_cfg(
                repo_id="lerobot/nyu_rot_dataset",
                camera_map={"primary": "observation.images.image"},
                weight=0.5,
                val_episode_count=2,
            ),
            _source_cfg(
                repo_id="lerobot/asu_table_top",
                camera_map={"primary": "observation.images.image"},
                weight=0.5,
                val_episode_count=5,
            ),
        ],
        request=_request_cfg(image_requests={"primary": [0, 1]}),
        loader=_loader_cfg(batch_size=4),
        adapter=_adapter_cfg(steps_per_epoch=8),
        output_format="raw",
    )
    dm.setup()

    seen = Counter()
    for batch in dm.train_dataloader():
        assert batch.meta is not None
        seen.update(batch.meta["source_name"])
        if len(seen) == 2:
            break

    assert set(seen) == {"lerobot/nyu_rot_dataset", "lerobot/asu_table_top"}


def test_real_multicamera_source_returns_multiple_image_streams() -> None:
    dm = LeRobotV3DataModule(
        sources=[
            _source_cfg(
                repo_id="lerobot/cmu_franka_exploration_dataset",
                camera_map={
                    "primary": "observation.images.image",
                    "secondary": "observation.images.highres_image",
                },
                val_episode_count=10,
            )
        ],
        request=_request_cfg(
            image_requests={
                "primary": [0, 1],
                "secondary": [0, 1],
            },
            include_actions=False,
        ),
        loader=_loader_cfg(batch_size=2),
        adapter=_adapter_cfg(steps_per_epoch=2),
        output_format="raw",
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert batch.image_streams is not None
    assert set(batch.image_streams) == {"primary", "secondary"}
    assert tuple(batch.image_streams["primary"].shape[:3]) == (2, 2, 3)
    assert tuple(batch.image_streams["secondary"].shape[:3]) == (2, 2, 3)
    assert batch.image_padding_masks is not None
    assert torch.all(batch.image_padding_masks["primary"])
