from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from stage2.callbacks import (
    PolicyTrainSampleVizConfig,
    PolicyTrainSampleVisualizationCallback,
)
from stage2.backends.interfaces import BackendMode, Stage2Batch, LatentOutput


class DummyCodeProvider:
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        b = video.shape[0]
        return torch.tensor([[3, 1, 7, 0]] * b, dtype=torch.long)


class DummyBackend:
    def latent_from_batch(
        self, batch: Stage2Batch, *, mode: BackendMode
    ) -> LatentOutput:
        assert mode is BackendMode.CODES
        assert batch.image_streams is not None
        assert batch.image_padding_masks is not None
        assert batch.state is not None
        b = int(batch.image_streams["observation.images.rgb"].shape[0])
        mask = batch.image_padding_masks["observation.images.rgb"]
        assert mask.dtype == torch.bool
        assert tuple(mask.shape) == (b,)
        assert torch.equal(mask, torch.ones((b,), dtype=torch.bool, device=mask.device))
        assert tuple(batch.state.shape) == (b, 2)
        tokens = torch.tensor([[3, 1, 7, 0]] * b, dtype=torch.long)
        return LatentOutput(tokens=tokens, logits=None, vector=None, meta=None)


def test_policy_train_visualization_callback_writes_files(tmp_path):
    callback = PolicyTrainSampleVisualizationCallback(
        PolicyTrainSampleVizConfig(enabled=True, num_samples=2, every_n_steps=10)
    )

    trainer = SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=10,
        logger=False,
    )

    pl_module = SimpleNamespace(
        action_tokens=SimpleNamespace(
            format_target=lambda codes: f"<ACTION> {codes} </ACTION>"
        ),
        backend=DummyBackend(),
        frames_to_images=lambda frames: [
            Image.new("RGB", (16, 16), color=(10, 20, 30))
            for _ in range(frames.shape[0])
        ],
        code_provider=DummyCodeProvider(),
        backend_mode=BackendMode.CODES,
        normalization_stats=None,
        _extract_policy_image_streams=lambda frames: {
            "observation.images.rgb": frames[:, 0, ...]
        },
        _extract_policy_image_padding_masks=lambda image_streams: {
            "observation.images.rgb": torch.ones(
                (int(image_streams["observation.images.rgb"].shape[0]),),
                dtype=torch.bool,
            )
        },
    )

    batch = Stage2Batch(
        image_streams={
            "primary": torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8)
        },
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick up block", "push button"],
        state=torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32),
        meta={
            "episode_id": ["ep1", "ep2"],
            "frame_idx": [0, 5],
        },
    )

    callback.on_train_batch_end(
        trainer, pl_module, outputs=None, batch=batch, batch_idx=0
    )

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "train_samples_step000010.png").exists()
    assert (viz_dir / "train_samples_step000010.json").exists()
