from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import pytest

from foundation.backends.interfaces import BackendMode, FoundationBatch, LatentOutput, LossOutput
from foundation.vla_backend_module import VLATokenBackendLightningModule, VLAOptimizerConfig
from common.lerobot_v3_types import Stage1Batch
from laq.task import LAQTask
from omegaconf import OmegaConf


class _CaptureBackend:
    codebook_size = 8
    code_seq_len = 1

    def __init__(self) -> None:
        self.last_batch: FoundationBatch | None = None

    def setup(self, *, device: torch.device) -> None:
        del device

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        del mode
        self.last_batch = batch
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
        assert tuple(video.shape[:3]) == (2, 3, 2)
        return torch.zeros((2, 1), dtype=torch.long)

    def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert tuple(video.shape[:3]) == (2, 3, 2)
        return torch.zeros((2, 1), dtype=torch.long), torch.zeros((2, 1, self.codebook_dim), dtype=torch.float32)


def test_vla_module_accepts_foundation_batch_inputs() -> None:
    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=_DummyCodeProvider(),
        backend_mode=BackendMode.ACTIONS,
        normalization_stats={
            "action": {"mean": [1.0, 2.0], "std": [2.0, 2.0]},
            "observation.state": {"mean": [1.0, -1.0], "std": [2.0, 4.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[3.0, 3.0]], [[1.0, -1.0]]], dtype=torch.float32),
        target_actions=torch.tensor(
            [[[3.0, 6.0]], [[1.0, 2.0]]],
            dtype=torch.float32,
        ),
        action_is_pad=torch.zeros((2, 1), dtype=torch.bool),
    )

    module._loss_and_targets_from_batch(batch)

    assert backend.last_batch is not None
    assert torch.allclose(
        backend.last_batch.state,
        torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=torch.float32),
    )
    assert torch.allclose(
        backend.last_batch.target_actions,
        torch.tensor(
            [[[1.0, 2.0]], [[0.0, 0.0]]],
            dtype=torch.float32,
        ),
    )


def test_vla_module_actions_mode_does_not_require_code_provider() -> None:
    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=None,
        backend_mode=BackendMode.ACTIONS,
        normalization_stats={
            "action": {"mean": [1.0, 2.0], "std": [2.0, 2.0]},
            "observation.state": {"mean": [1.0, -1.0], "std": [2.0, 4.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[3.0, 3.0]], [[1.0, -1.0]]], dtype=torch.float32),
        target_actions=torch.tensor(
            [[[3.0, 6.0]], [[1.0, 2.0]]],
            dtype=torch.float32,
        ),
        action_is_pad=torch.zeros((2, 1), dtype=torch.bool),
    )

    module._loss_and_targets_from_batch(batch)

    assert backend.last_batch is not None
    assert torch.allclose(
        backend.last_batch.state,
        torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=torch.float32),
    )
    assert torch.allclose(
        backend.last_batch.target_actions,
        torch.tensor(
            [[[1.0, 2.0]], [[0.0, 0.0]]],
            dtype=torch.float32,
        ),
    )


def test_vla_module_latent_flow_requires_code_provider() -> None:
    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=None,
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[3.0, 3.0]], [[1.0, -1.0]]], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="code_provider"):
        module._loss_and_targets_from_batch(batch)


def test_laq_task_accepts_stage1_batch() -> None:
    model_config = OmegaConf.create(
        {
            "dim": 128,
            "quant_dim": 8,
            "codebook_size": 8,
            "image_size": 256,
            "patch_size": 32,
            "spatial_depth": 1,
            "temporal_depth": 1,
            "dim_head": 16,
            "heads": 2,
            "code_seq_len": 2,
            "channels": 3,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "vq_discarding_threshold": 0.1,
            "vq_discarding_threshold_schedule": None,
            "latent_ablation": "none",
            "use_dinov3_encoder": False,
            "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3_pool_to_grid": None,
            "dino": {"enabled": False},
            "use_pixel_decoder": True,
            "use_aux_decoder": False,
            "flow": {"enabled": False},
            "codebook_replace_schedule": [[10, 100]],
        }
    )
    training_config = OmegaConf.create(
        {
            "max_steps": 10,
            "metrics": {"log_every_n_steps": 1, "num_unique_codes_every_n_steps": 1},
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0.01, "eps": 1e-8},
            "scheduler": {"type": "cosine", "min_lr": 1e-6, "warmup_steps": 0, "warmup_start_lr": 1e-6},
            "gradient": {"clip_val": 1.0},
        }
    )

    task = LAQTask(model_config=model_config, training_config=training_config)
    batch = Stage1Batch(
        image_streams={"primary": torch.randn(2, 2, 3, 256, 256)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        meta={"dataset_name": ["a", "b"]},
    )

    loss = task.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_laq_task_transfer_batch_to_device_accepts_stage1_batch() -> None:
    model_config = OmegaConf.create(
        {
            "dim": 128,
            "quant_dim": 8,
            "codebook_size": 8,
            "image_size": 256,
            "patch_size": 32,
            "spatial_depth": 1,
            "temporal_depth": 1,
            "dim_head": 16,
            "heads": 2,
            "code_seq_len": 2,
            "channels": 3,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "vq_discarding_threshold": 0.1,
            "vq_discarding_threshold_schedule": None,
            "latent_ablation": "none",
            "use_dinov3_encoder": False,
            "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3_pool_to_grid": None,
            "dino": {"enabled": False},
            "use_pixel_decoder": True,
            "use_aux_decoder": False,
            "flow": {"enabled": False},
            "codebook_replace_schedule": [[10, 100]],
        }
    )
    training_config = OmegaConf.create(
        {
            "max_steps": 10,
            "metrics": {"log_every_n_steps": 1, "num_unique_codes_every_n_steps": 1},
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0.01, "eps": 1e-8},
            "scheduler": {"type": "cosine", "min_lr": 1e-6, "warmup_steps": 0, "warmup_start_lr": 1e-6},
            "gradient": {"clip_val": 1.0},
        }
    )

    task = LAQTask(model_config=model_config, training_config=training_config)
    batch = Stage1Batch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 32, 32), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        meta={"dataset_name": ["a", "b"]},
    )

    moved = task.transfer_batch_to_device(batch, device=torch.device("cpu"), dataloader_idx=0)

    assert isinstance(moved, Stage1Batch)
    assert moved.image_streams["primary"].dtype == torch.float32
    assert float(moved.image_streams["primary"].min().item()) >= 0.0
    assert float(moved.image_streams["primary"].max().item()) <= 1.0


def test_vla_module_converts_foundation_uint8_video_for_online_laq() -> None:
    class _CaptureCodeProvider(_DummyCodeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.last_video: torch.Tensor | None = None

        def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.last_video = video
            return super().codes_and_vectors_from_video(video)

    backend = _CaptureBackend()
    code_provider = _CaptureCodeProvider()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=code_provider,
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32),
    )

    module._loss_and_targets_from_batch(batch)

    assert code_provider.last_video is not None
    assert code_provider.last_video.dtype == torch.float32
    assert float(code_provider.last_video.min().item()) >= 0.0
    assert float(code_provider.last_video.max().item()) <= 1.0


def test_vla_module_resizes_online_laq_video_to_teacher_image_size() -> None:
    class _CaptureResizeCodeProvider(_DummyCodeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.last_video: torch.Tensor | None = None
            self.image_size = (4, 4)

        def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.last_video = video
            return super().codes_and_vectors_from_video(video)

    backend = _CaptureBackend()
    code_provider = _CaptureResizeCodeProvider()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=code_provider,
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32),
    )

    module._loss_and_targets_from_batch(batch)

    assert code_provider.last_video is not None
    assert tuple(code_provider.last_video.shape) == (2, 3, 2, 4, 4)


def test_vla_module_validation_step_accepts_foundation_batch() -> None:
    class _CaptureCodeProvider(_DummyCodeProvider):
        def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return super().codes_and_vectors_from_video(video)

    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=_CaptureCodeProvider(),
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [1.0, -1.0], "std": [2.0, 4.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[3.0, 3.0]], [[1.0, -1.0]]], dtype=torch.float32),
    )

    loss = module.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert backend.last_batch is not None
    assert torch.allclose(
        backend.last_batch.state,
        torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=torch.float32),
    )


def test_vla_module_validation_step_reads_foundation_batch_metadata() -> None:
    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=_DummyCodeProvider(),
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )
    module.log = lambda *args, **kwargs: None
    module.__dict__["_trainer"] = SimpleNamespace(global_step=7)

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32),
        meta={
            "dataset_name": ["ds_a", "ds_b"],
            "episode_id": [11, 12],
            "frame_idx": [21, 22],
        },
    )

    module.reset_val_batch_payload_queue()
    loss = module.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    payload = module.consume_next_val_batch_payload()
    assert isinstance(payload, dict)
    records = payload.get("records")
    assert isinstance(records, list)
    assert len(records) == 2
    assert records[0]["metadata"]["dataset_name"] == "ds_a"
    assert records[1]["metadata"]["dataset_name"] == "ds_b"
    assert records[0]["metadata"]["episode_id"] == 11
    assert records[1]["metadata"]["episode_id"] == 12
    assert records[0]["metadata"]["frame_idx"] == 21
    assert records[1]["metadata"]["frame_idx"] == 22


def test_vla_module_validation_step_enqueues_payload_for_all_batches() -> None:
    backend = _CaptureBackend()
    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=_DummyCodeProvider(),
        backend_mode=BackendMode.LATENT_FLOW,
        normalization_stats={
            "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        optimizer=VLAOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )
    module.log = lambda *args, **kwargs: None
    module.__dict__["_trainer"] = SimpleNamespace(global_step=13)

    batch = FoundationBatch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32),
        meta={
            "dataset_name": ["ds_a", "ds_b"],
            "episode_id": [11, 12],
            "frame_idx": [21, 22],
        },
    )

    module.reset_val_batch_payload_queue()
    module.validation_step(batch, batch_idx=0)
    module.validation_step(batch, batch_idx=1)

    payload0 = module.consume_next_val_batch_payload()
    payload1 = module.consume_next_val_batch_payload()
    payload2 = module.consume_next_val_batch_payload()

    assert isinstance(payload0, dict)
    assert isinstance(payload1, dict)
    assert payload2 is None
    assert isinstance(payload0.get("records"), list)
    assert isinstance(payload1.get("records"), list)
    assert len(payload0["records"]) == 2
    assert len(payload1["records"]) == 2
