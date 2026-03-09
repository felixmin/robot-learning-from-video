from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")
import lightning.pytorch as pl

from stage2.callbacks import ThroughputLoggingCallback, ThroughputLoggingConfig


class DummyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.logged = []

    def log(self, name, value, **kwargs):  # type: ignore[override]
        self.logged.append((name, float(value), kwargs))


def test_throughput_callback_logs_steps_per_sec(monkeypatch):
    times = iter([100.0, 100.5])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))

    callback = ThroughputLoggingCallback(
        ThroughputLoggingConfig(enabled=True, log_every_n_steps=5)
    )
    module = DummyModule()
    trainer = SimpleNamespace(global_step=0)

    callback.on_train_start(trainer, module)

    # Simulate reaching global_step=5 at batch end.
    trainer.global_step = 5
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch={"frames": SimpleNamespace(shape=(2,))},
        batch_idx=0,
    )

    keys = [k for (k, _v, _kw) in module.logged]
    assert "perf/steps_per_sec" in keys
    assert "perf/samples_per_sec" in keys
