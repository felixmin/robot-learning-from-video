from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("lightning")

from stage2.validation_checks import CHECK_TYPES, Stage2ValidationCheck
from stage2.validation_pipeline_callback import Stage2ValidationPipelineCallback


class _DummyModule:
    def __init__(self) -> None:
        self.payload_queue: list[dict] = []
        self.reset_calls = 0
        self.logged_dicts: list[dict[str, float]] = []
        self.action_tokens = SimpleNamespace(
            codebook_size=16,
            format_target=lambda codes: f"<ACTION> {codes} </ACTION>",
        )

    def enqueue_payload(self, payload: dict) -> None:
        self.payload_queue.append(payload)

    def consume_next_val_batch_payload(self):
        if not self.payload_queue:
            return None
        return self.payload_queue.pop(0)

    def reset_val_batch_payload_queue(self) -> None:
        self.payload_queue.clear()
        self.reset_calls += 1

    def log_dict(self, metrics: dict[str, float], sync_dist: bool = True) -> None:
        del sync_dist
        self.logged_dicts.append(dict(metrics))

    def frames_to_images(self, frames: torch.Tensor):
        return [
            Image.new("RGB", (16, 16), color=(10, 20, 30))
            for _ in range(int(frames.shape[0]))
        ]


class _BoomCheck(Stage2ValidationCheck):
    def run(
        self,
        cache,
        pl_module,
        trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, float]:
        del cache, pl_module, trainer, metric_suffix
        raise RuntimeError("boom")


class _NoOutputCheck(Stage2ValidationCheck):
    def run(
        self,
        cache,
        pl_module,
        trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, float]:
        del cache, pl_module, trainer, metric_suffix
        return {"_produced": 0, "_reason": "test_no_output"}


def _make_record(
    *, dataset_name: str = "mix", gt_codes: list[int] | None = None
) -> dict:
    return {
        "mode": "codes",
        "frame": torch.randint(0, 256, (2, 16, 16, 3), dtype=torch.uint8),
        "instruction": "pick block",
        "gt_codes": gt_codes if gt_codes is not None else [1, 2, 3, 4],
        "pred_codes": [1, 2, 3, 4],
        "gt_vector": None,
        "pred_vector": None,
        "gt_action": None,
        "pred_action": None,
        "gen_debug": None,
        "metadata": {
            "dataset_name": dataset_name,
            "episode_id": "ep",
            "frame_idx": 0,
        },
    }


def _trainer(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        is_global_zero=True,
        default_root_dir=str(tmp_path),
        global_step=11,
        logger=False,
    )


def test_validation_epoch_start_clears_cache_and_resets_module_queue(tmp_path):
    callback = Stage2ValidationPipelineCallback(checks_config={})
    callback.global_cache.add_record(_make_record())

    module = _DummyModule()
    module.enqueue_payload({"records": [_make_record()]})

    callback.on_validation_epoch_start(_trainer(tmp_path), module)

    assert callback.global_cache.sample_count() == 0
    assert module.reset_calls == 1
    assert module.payload_queue == []


def test_batch_idx_gt_zero_payload_is_accumulated_across_batches(tmp_path):
    callback = Stage2ValidationPipelineCallback(
        checks_config={
            "dist": {
                "type": "token_distribution",
                "enabled": True,
                "min_samples": 1,
            }
        }
    )
    module = _DummyModule()
    trainer = _trainer(tmp_path)

    callback.on_validation_epoch_start(trainer, module)
    module.enqueue_payload({"records": [_make_record(gt_codes=[1, 1, 1, 1])]})
    module.enqueue_payload({"records": [_make_record(gt_codes=[2, 2, 2, 2])]})

    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )
    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=1
    )

    assert callback.global_cache.sample_count() == 2
    all_codes = callback.global_cache.get_all_gt_codes()
    assert isinstance(all_codes, torch.Tensor)
    assert tuple(all_codes.shape) == (2, 4)


def test_bucket_routing_applies_metric_suffix(tmp_path):
    callback = Stage2ValidationPipelineCallback(
        checks_config={
            "dist": {
                "type": "token_distribution",
                "enabled": True,
                "buckets": ["focus"],
            }
        },
        bucket_configs={
            "focus": {
                "filters": {"dataset_name": "focus_ds"},
            }
        },
    )
    module = _DummyModule()
    trainer = _trainer(tmp_path)

    callback.on_validation_epoch_start(trainer, module)
    module.enqueue_payload(
        {
            "records": [
                _make_record(dataset_name="focus_ds", gt_codes=[1, 2, 3, 4]),
                _make_record(dataset_name="other_ds", gt_codes=[5, 6, 7, 8]),
            ]
        }
    )
    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )
    callback.on_validation_epoch_end(trainer, module)

    merged = {}
    for item in module.logged_dicts:
        merged.update(item)

    assert "val/token_distribution_entropy_focus" in merged
    assert "val/token_distribution_utilization_focus" in merged


def test_pipeline_startup_rejects_unknown_check_type():
    with pytest.raises(ValueError, match="Unknown Stage 2 validation check type"):
        Stage2ValidationPipelineCallback(
            checks_config={
                "unknown": {
                    "type": "not_a_check",
                }
            }
        )


def test_pipeline_startup_rejects_unknown_bucket_binding():
    with pytest.raises(ValueError, match="unknown bucket"):
        Stage2ValidationPipelineCallback(
            checks_config={
                "dist": {
                    "type": "token_distribution",
                    "buckets": ["missing_bucket"],
                }
            },
            bucket_configs={},
        )


def test_pipeline_startup_requires_lam_checkpoint_for_enabled_flow_decode():
    with pytest.raises(ValueError, match="requires non-empty lam_checkpoint_path"):
        Stage2ValidationPipelineCallback(
            checks_config={
                "flow": {
                    "type": "latent_flow_decode",
                    "enabled": True,
                    "lam_checkpoint_path": "",
                }
            }
        )


def test_pipeline_soft_fail_continues_running_remaining_checks(tmp_path):
    original = CHECK_TYPES.get("boom")
    CHECK_TYPES["boom"] = _BoomCheck
    try:
        callback = Stage2ValidationPipelineCallback(
            checks_config={
                "boom_one": {
                    "type": "boom",
                    "enabled": True,
                },
                "dist": {
                    "type": "token_distribution",
                    "enabled": True,
                },
            }
        )
        module = _DummyModule()
        trainer = _trainer(tmp_path)

        callback.on_validation_epoch_start(trainer, module)
        module.enqueue_payload({"records": [_make_record(gt_codes=[1, 2, 3, 4])]})
        callback.on_validation_batch_end(
            trainer, module, outputs=None, batch=None, batch_idx=0
        )

        callback.on_validation_epoch_end(trainer, module)

        merged = {}
        for item in module.logged_dicts:
            merged.update(item)
        assert "val/token_distribution_entropy" in merged
    finally:
        if original is None:
            CHECK_TYPES.pop("boom", None)
        else:
            CHECK_TYPES["boom"] = original


def test_sample_panels_check_writes_artifacts(tmp_path):
    callback = Stage2ValidationPipelineCallback(
        checks_config={
            "samples": {
                "type": "sample_panels",
                "enabled": True,
                "num_samples": 1,
            }
        }
    )
    module = _DummyModule()
    trainer = _trainer(tmp_path)

    callback.on_validation_epoch_start(trainer, module)
    module.enqueue_payload({"records": [_make_record(gt_codes=[3, 1, 7, 0])]})
    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )
    callback.on_validation_epoch_end(trainer, module)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "val_samples_step000011.png").exists()
    assert (viz_dir / "val_samples_step000011.json").exists()


def test_sample_panels_check_writes_artifacts_without_backend_image_adapter(tmp_path):
    callback = Stage2ValidationPipelineCallback(
        checks_config={
            "samples": {
                "type": "sample_panels",
                "enabled": True,
                "num_samples": 1,
            }
        }
    )
    module = _DummyModule()
    module.frames_to_images = None
    trainer = _trainer(tmp_path)

    callback.on_validation_epoch_start(trainer, module)
    module.enqueue_payload({"records": [_make_record(gt_codes=[3, 1, 7, 0])]})
    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )
    callback.on_validation_epoch_end(trainer, module)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "val_samples_step000011.png").exists()
    assert (viz_dir / "val_samples_step000011.json").exists()


def test_sample_panels_check_writes_artifacts_when_adapter_returns_tensor(tmp_path):
    callback = Stage2ValidationPipelineCallback(
        checks_config={
            "samples": {
                "type": "sample_panels",
                "enabled": True,
                "num_samples": 1,
            }
        }
    )
    module = _DummyModule()
    module.frames_to_images = lambda frames: torch.rand(
        (int(frames.shape[0]), 3, 16, 16), dtype=torch.float32
    )
    trainer = _trainer(tmp_path)

    callback.on_validation_epoch_start(trainer, module)
    module.enqueue_payload({"records": [_make_record(gt_codes=[3, 1, 7, 0])]})
    callback.on_validation_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )
    callback.on_validation_epoch_end(trainer, module)

    viz_dir = tmp_path / "visualizations"
    assert (viz_dir / "val_samples_step000011.png").exists()
    assert (viz_dir / "val_samples_step000011.json").exists()


def test_pipeline_counts_no_output_result_as_skip(tmp_path, caplog):
    original = CHECK_TYPES.get("no_output")
    CHECK_TYPES["no_output"] = _NoOutputCheck
    try:
        callback = Stage2ValidationPipelineCallback(
            checks_config={
                "noop": {
                    "type": "no_output",
                    "enabled": True,
                    "min_samples": 1,
                }
            }
        )
        module = _DummyModule()
        trainer = _trainer(tmp_path)

        callback.on_validation_epoch_start(trainer, module)
        module.enqueue_payload({"records": [_make_record(gt_codes=[1, 2, 3, 4])]})
        callback.on_validation_batch_end(
            trainer, module, outputs=None, batch=None, batch_idx=0
        )
        with caplog.at_level(logging.INFO, logger="stage2.training"):
            callback.on_validation_epoch_end(trainer, module)

        text = caplog.text
        assert "[Stage2Validation] due=1 ran=0 skipped=1 soft_failed=0" in text
        assert "[Stage2Validation] skip noop: test_no_output" in text
    finally:
        if original is None:
            CHECK_TYPES.pop("no_output", None)
        else:
            CHECK_TYPES["no_output"] = original
