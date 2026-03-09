from __future__ import annotations

from types import SimpleNamespace

from common.callbacks import ProgressLoggerCallback


def test_progress_logger_callback_prints_every_n_steps(capsys):
    cb = ProgressLoggerCallback(log_every_n_steps=2)
    trainer = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[])

    # step=0 -> (0+1)%2 != 0 => no print
    cb.on_train_batch_end(trainer, None, outputs={"loss": 1.0}, batch=None, batch_idx=0)
    assert capsys.readouterr().out == ""

    # step=1 -> (1+1)%2 == 0 => print
    trainer.global_step = 1
    cb.on_train_batch_end(trainer, None, outputs={"loss": 1.0}, batch=None, batch_idx=0)
    out = capsys.readouterr().out
    assert "[Step 2]" in out
    assert "loss=" in out


def test_progress_logger_callback_prints_val_metrics(capsys):
    cb = ProgressLoggerCallback(log_every_n_steps=1)
    trainer = SimpleNamespace(
        global_step=5,
        callback_metrics={"val/loss": 0.25, "train/loss": 1.0},
    )
    cb.on_validation_end(trainer, None)
    out = capsys.readouterr().out
    assert "[Validation]" in out
    assert "val/loss=" in out
