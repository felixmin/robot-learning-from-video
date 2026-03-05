from __future__ import annotations

from types import SimpleNamespace

import torch

from common.callbacks import DataSampleVisualizationCallback, DataSampleVisualizationConfig
from common.lerobot_v3_types import Stage1Batch
from stage2.backends.interfaces import Stage2Batch


class _DummyImageLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_image(self, *, key, images, caption):
        self.calls.append({"key": key, "images": images, "caption": caption})


def test_data_sample_visualization_callback_stage1_logs_frame_pairs() -> None:
    logger = _DummyImageLogger()
    trainer = SimpleNamespace(loggers=[logger], is_global_zero=True, global_step=10)
    callback = DataSampleVisualizationCallback(
        DataSampleVisualizationConfig(
            enabled=True,
            every_n_steps=10,
            num_samples=2,
            key="data/train_samples",
            mode="stage1",
        )
    )
    batch = Stage1Batch(
        image_streams={"primary": torch.randint(0, 256, (3, 2, 8, 8, 3), dtype=torch.uint8)},
        task_text=["pick", "place", "push"],
        meta={
            "dataset_short": ["bridge", "language_table", "rt1"],
            "episode_id": [1, 2, 3],
            "frame_idx": [10, 20, 30],
        },
    )

    callback.on_train_batch_end(trainer, pl_module=None, outputs=None, batch=batch, batch_idx=0)

    assert len(logger.calls) == 1
    call = logger.calls[0]
    assert call["key"] == "data/train_samples"
    images = call["images"]
    assert isinstance(images, list) and len(images) == 1
    grid = images[0]
    assert isinstance(grid, torch.Tensor)
    assert tuple(grid.shape)[0] == 3


def test_data_sample_visualization_callback_stage2_logs_single_frames() -> None:
    logger = _DummyImageLogger()
    trainer = SimpleNamespace(loggers=[logger], is_global_zero=True, global_step=20)
    callback = DataSampleVisualizationCallback(
        DataSampleVisualizationConfig(
            enabled=True,
            every_n_steps=10,
            num_samples=3,
            key="data/train_samples",
            mode="stage2",
        )
    )
    batch = Stage2Batch(
        image_streams={"primary": torch.randint(0, 256, (4, 2, 8, 8, 3), dtype=torch.uint8)},
        task_text=["pick", "place", "push", "pull"],
        meta={
            "dataset_short": ["bridge", "language_table", "rt1", "roboturk"],
            "episode_id": [1, 2, 3, 4],
            "frame_idx": [10, 20, 30, 40],
        },
    )

    callback.on_train_batch_end(trainer, pl_module=None, outputs=None, batch=batch, batch_idx=0)

    assert len(logger.calls) == 1
    call = logger.calls[0]
    assert call["key"] == "data/train_samples"
    caption = call["caption"]
    assert isinstance(caption, list) and len(caption) == 1
    assert "task=pick" in str(caption[0])
