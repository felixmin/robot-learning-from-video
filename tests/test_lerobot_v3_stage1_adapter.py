from __future__ import annotations

import torch

from common.lerobot_v3_adapters import dataset_batch_to_stage1_batch
from common.lerobot_v3_types import BatchedDatasetSample


def _batched_sample(time_steps: int = 2) -> BatchedDatasetSample:
    return BatchedDatasetSample(
        image_streams={
            "primary": torch.arange(
                2 * time_steps * 3 * 4 * 4, dtype=torch.uint8
            ).reshape(2, time_steps, 3, 4, 4),
            "wrist": torch.ones((2, time_steps, 3, 4, 4), dtype=torch.uint8),
        },
        image_padding_masks={
            "primary": torch.zeros((2, time_steps), dtype=torch.bool),
            "wrist": torch.zeros((2, time_steps), dtype=torch.bool),
        },
        state=torch.ones((2, 1, 3), dtype=torch.float32),
        state_is_pad=torch.zeros((2, 1), dtype=torch.bool),
        action=torch.ones((2, 4, 2), dtype=torch.float32),
        action_is_pad=torch.zeros((2, 4), dtype=torch.bool),
        task_text=["pick", "place"],
        meta={"dataset_name": ["a", "b"], "frame_idx": [10, 20]},
    )


def test_dataset_batch_to_stage1_batch_preserves_multicamera_temporal_shape() -> None:
    batch = dataset_batch_to_stage1_batch(_batched_sample())
    assert tuple(batch.image_streams["primary"].shape) == (2, 2, 3, 4, 4)
    assert tuple(batch.image_streams["wrist"].shape) == (2, 2, 3, 4, 4)


def test_dataset_batch_to_stage1_batch_handles_optional_action_and_state() -> None:
    batch = dataset_batch_to_stage1_batch(_batched_sample())
    assert tuple(batch.state.shape) == (2, 1, 3)
    assert tuple(batch.action.shape) == (2, 4, 2)
    assert batch.task_text == ["pick", "place"]
