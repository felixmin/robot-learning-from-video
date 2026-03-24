from __future__ import annotations

import logging

from common.unified_logging import setup_wandb_with_unified_paths


def test_setup_wandb_with_unified_paths_uses_explicit_wandb_run(
    monkeypatch, tmp_path
) -> None:
    calls: dict[str, object] = {}
    fake_run = object()

    def fake_init(**kwargs):
        calls["init_kwargs"] = dict(kwargs)
        return fake_run

    class _FakeWandbLogger:
        def __init__(self, **kwargs):
            calls["logger_kwargs"] = dict(kwargs)

    monkeypatch.setattr("wandb.init", fake_init)
    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", _FakeWandbLogger)

    logger = logging.getLogger("test.unified_logging")
    wandb_logger = setup_wandb_with_unified_paths(
        logger=logger,
        output_dir=tmp_path,
        project="hlrp",
        name="stage1_local",
        tags=["stage1"],
        use_wandb=True,
        group="stage1_local",
    )

    assert isinstance(wandb_logger, _FakeWandbLogger)
    assert calls["init_kwargs"] == {
        "project": "hlrp",
        "name": "stage1_local",
        "dir": str(tmp_path / "wandb"),
        "tags": ["stage1"],
        "reinit": "finish_previous",
        "group": "stage1_local",
    }
    assert calls["logger_kwargs"] == {
        "project": "hlrp",
        "name": "stage1_local",
        "save_dir": str(tmp_path / "wandb"),
        "tags": ["stage1"],
        "experiment": fake_run,
        "group": "stage1_local",
    }
