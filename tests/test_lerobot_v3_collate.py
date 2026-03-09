from __future__ import annotations

import torch
import pytest

from common.lerobot_v3_data import collate_dataset_samples
from common.lerobot_v3_types import DatasetSample


def _sample(*, offset: int = 0) -> DatasetSample:
    return DatasetSample(
        image_streams={
            "primary": torch.full((2, 3, 8, 8), offset + 1, dtype=torch.uint8),
            "wrist": torch.full((2, 3, 8, 8), offset + 2, dtype=torch.uint8),
        },
        image_padding_masks={
            "primary": torch.tensor([False, False], dtype=torch.bool),
            "wrist": torch.tensor([False, True], dtype=torch.bool),
        },
        state=torch.full((1, 4), float(offset), dtype=torch.float32),
        state_is_pad=torch.tensor([False], dtype=torch.bool),
        action=torch.full((3, 2), float(offset + 1), dtype=torch.float32),
        action_is_pad=torch.tensor([False, False, True], dtype=torch.bool),
        task_text=f"task-{offset}",
        meta={"dataset_name": f"ds-{offset}", "frame_idx": offset},
    )


def test_collate_dataset_samples_stacks_multicamera_temporal_tensors() -> None:
    batch = collate_dataset_samples([_sample(offset=0), _sample(offset=1)])
    assert tuple(batch.image_streams["primary"].shape) == (2, 2, 3, 8, 8)
    assert tuple(batch.image_padding_masks["wrist"].shape) == (2, 2)
    assert tuple(batch.state.shape) == (2, 1, 4)
    assert tuple(batch.action.shape) == (2, 3, 2)


def test_collate_dataset_samples_handles_missing_optional_fields() -> None:
    out = collate_dataset_samples(
        [
            DatasetSample(
                image_streams={"primary": torch.zeros((1, 3, 4, 4), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.zeros((1,), dtype=torch.bool)},
            ),
            DatasetSample(
                image_streams={"primary": torch.ones((1, 3, 4, 4), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.zeros((1,), dtype=torch.bool)},
            ),
        ]
    )
    assert out.state is None
    assert out.action is None
    assert out.task_text is None


def test_collate_dataset_samples_preserves_string_lists_and_meta_lists() -> None:
    batch = collate_dataset_samples([_sample(offset=0), _sample(offset=1)])
    assert batch.task_text == ["task-0", "task-1"]
    assert batch.meta == {
        "dataset_name": ["ds-0", "ds-1"],
        "frame_idx": [0, 1],
    }


def test_collate_dataset_samples_rejects_mismatched_camera_sets() -> None:
    with pytest.raises(ValueError, match="camera"):
        collate_dataset_samples(
            [
                DatasetSample(
                    image_streams={
                        "primary": torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
                    },
                    image_padding_masks={
                        "primary": torch.zeros((1,), dtype=torch.bool)
                    },
                ),
                DatasetSample(
                    image_streams={
                        "wrist": torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
                    },
                    image_padding_masks={"wrist": torch.zeros((1,), dtype=torch.bool)},
                ),
            ]
        )


def test_collate_dataset_samples_pads_state_and_action_dims_to_batch_max() -> None:
    out = collate_dataset_samples(
        [
            DatasetSample(
                image_streams={"primary": torch.zeros((1, 3, 4, 4), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.zeros((1,), dtype=torch.bool)},
                state=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                state_is_pad=torch.tensor([False], dtype=torch.bool),
                action=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                action_is_pad=torch.tensor([False, True], dtype=torch.bool),
            ),
            DatasetSample(
                image_streams={"primary": torch.ones((1, 3, 4, 4), dtype=torch.uint8)},
                image_padding_masks={"primary": torch.zeros((1,), dtype=torch.bool)},
                state=torch.tensor([[5.0, 6.0, 7.0]], dtype=torch.float32),
                state_is_pad=torch.tensor([False], dtype=torch.bool),
                action=torch.tensor(
                    [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                    dtype=torch.float32,
                ),
                action_is_pad=torch.tensor([False, False], dtype=torch.bool),
            ),
        ]
    )

    assert tuple(out.state.shape) == (2, 1, 3)
    assert torch.equal(
        out.state[0, 0], torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32)
    )
    assert tuple(out.action.shape) == (2, 2, 4)
    assert torch.equal(
        out.action[0, 0], torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float32)
    )
