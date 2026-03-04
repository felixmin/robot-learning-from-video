from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from foundation.backends.interfaces import BackendMode, FoundationBatch, LatentOutput, LossOutput
from foundation.vla_backend_module import VLATokenBackendLightningModule, VLAOptimizerConfig


class _CaptureBackend:
    codebook_size = 8
    code_seq_len = 1

    def __init__(self) -> None:
        self.last_batch: FoundationBatch | None = None
        self.last_mode: BackendMode | None = None

    def setup(self, *, device: torch.device) -> None:
        del device

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        self.last_batch = batch
        self.last_mode = mode
        return LossOutput(loss=torch.tensor(0.0), metrics={"loss": 0.0})

    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        del batch, mode
        return LatentOutput()


@dataclass
class _DummyCodeProvider:
    codebook_size: int = 8
    code_seq_len: int = 1
    codebook_dim: int = 2

    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        del video
        raise RuntimeError("not used in ACTIONS mode")

    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        del codes
        raise RuntimeError("not used in ACTIONS mode")

    def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del video
        raise RuntimeError("not used in ACTIONS mode")


def test_stage2_module_applies_configured_normalization_stats() -> None:
    backend = _CaptureBackend()
    provider = _DummyCodeProvider()

    stats: dict[str, dict[str, Any]] = {
        "action": {"mean": [1.0, 2.0], "std": [2.0, 2.0]},
        "observation.state": {"mean": [1.0, -1.0], "std": [2.0, 4.0]},
    }

    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=provider,
        backend_mode=BackendMode.ACTIONS,
        normalization_stats=stats,
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={
            "observation.images.rgb": torch.randint(0, 255, (2, 2, 8, 8, 3), dtype=torch.uint8),
        },
        task_text=["pick", "place"],
        state=torch.tensor([[3.0, 3.0], [1.0, -1.0]], dtype=torch.float32),
        target_actions=torch.tensor([[3.0, 6.0], [1.0, 2.0]], dtype=torch.float32),
        action_is_pad=torch.zeros((2, 1), dtype=torch.bool),
    )

    module._loss_and_targets_from_batch(batch)

    assert backend.last_batch is not None
    assert backend.last_mode is BackendMode.ACTIONS

    expected_actions = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    expected_state = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    assert backend.last_batch.image_streams is not None
    assert "observation.images.rgb" in backend.last_batch.image_streams
    assert torch.allclose(backend.last_batch.target_actions, expected_actions)
    assert torch.allclose(backend.last_batch.state, expected_state)
