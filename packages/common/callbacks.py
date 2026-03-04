"""
Common Lightning callbacks shared across stages.
"""

from __future__ import annotations

import collections
from typing import Any, Optional

from lightning.pytorch.callbacks import Callback

def _batch_meta_list(batch: Any, *, key: str) -> list[Any] | None:
    if isinstance(batch, dict):
        items = batch.get(key)
        return items if isinstance(items, list) else None
    meta = getattr(batch, "meta", None)
    if meta is not None:
        if not isinstance(meta, dict):
            return None
        items = meta.get(key)
        return items if isinstance(items, list) else None
    return None


class ProgressLoggerCallback(Callback):
    """
    Log training progress to stdout for cluster jobs where tqdm doesn't work well
    in log files.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = int(log_every_n_steps)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.log_every_n_steps <= 0:
            return
        if (int(trainer.global_step) + 1) % self.log_every_n_steps != 0:
            return

        loss: Optional[float] = None
        try:
            loss_obj = outputs.get("loss") if isinstance(outputs, dict) else outputs
            loss = float(loss_obj) if loss_obj is not None else None
        except Exception:
            loss = None

        lr: Optional[float] = None
        try:
            if trainer.optimizers:
                lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = None

        # Use print() for progress output (captured by WandB and unified logging).
        msg = f"[Step {int(trainer.global_step) + 1}]"
        if loss is not None:
            msg += f" loss={loss:.4f},"
        if lr is not None:
            msg += f" lr={lr:.2e},"
        msg += f" epoch={int(trainer.current_epoch)}"
        print(msg)

    def on_validation_end(self, trainer, pl_module) -> None:
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "val" in k}
        if not metrics:
            return
        try:
            metrics_str = ", ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
        except Exception:
            return
        print(f"[Validation] step={int(trainer.global_step)}, {metrics_str}")


class DatasetUsageLoggerCallback(Callback):
    """
    Track and report how many samples from each dataset were seen.

    Expects dataset names in batch metadata (`dataset_name`) and reports counts since the
    last report. Can be aligned with step-based validation by printing on each validation end.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_on_validation_end: bool = True,
        log_every_n_steps: Optional[int] = None,
        log_batch_composition_every_n_steps: Optional[int] = None,
        key: str = "dataset_name",
        top_k: int = 12,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.log_on_validation_end = bool(log_on_validation_end)
        self.log_every_n_steps = int(log_every_n_steps) if log_every_n_steps else None
        self.log_batch_composition_every_n_steps = (
            int(log_batch_composition_every_n_steps)
            if log_batch_composition_every_n_steps
            else None
        )
        self.key = str(key)
        self.top_k = int(top_k)

        self._since_last: collections.Counter[str] = collections.Counter()
        self._total: collections.Counter[str] = collections.Counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int) -> None:
        if not self.enabled:
            return
        items = _batch_meta_list(batch, key=self.key)
        if items is None:
            return

        for x in items:
            name = str(x) if x is not None else "None"
            self._since_last[name] += 1
            self._total[name] += 1

        if self.log_every_n_steps is not None:
            step = int(getattr(trainer, "global_step", 0))
            if self.log_every_n_steps > 0 and (step + 1) % self.log_every_n_steps == 0:
                self._print_summary(trainer, prefix="Train", reset_since_last=True)

        if self.log_batch_composition_every_n_steps is not None:
            step = int(getattr(trainer, "global_step", 0))
            if self.log_batch_composition_every_n_steps > 0 and (step + 1) % self.log_batch_composition_every_n_steps == 0:
                self._print_batch_mix(step=step + 1, items=items)

    def on_validation_end(self, trainer, pl_module) -> None:
        if not self.enabled or not self.log_on_validation_end:
            return
        self._print_summary(trainer, prefix="Validation", reset_since_last=True)

    def _print_summary(self, trainer, *, prefix: str, reset_since_last: bool) -> None:
        if not self._since_last:
            return

        total = sum(self._since_last.values())
        parts = []
        for name, count in self._since_last.most_common(max(1, self.top_k)):
            pct = 100.0 * float(count) / float(total) if total > 0 else 0.0
            parts.append(f"{name}={count} ({pct:.1f}%)")

        step = int(getattr(trainer, "global_step", 0))
        msg = f"[{prefix}][DatasetUsage] step={step} interval_total={total} " + ", ".join(parts)
        print(msg)

        if reset_since_last:
            self._since_last.clear()

    def _print_batch_mix(self, *, step: int, items: list[Any]) -> None:
        if not items:
            return
        counts = collections.Counter(str(x) if x is not None else "None" for x in items)
        total = len(items)
        parts = []
        for name, count in counts.most_common(max(1, self.top_k)):
            pct = 100.0 * float(count) / float(total)
            parts.append(f"{name}={count} ({pct:.1f}%)")
        print(
            f"[Train][BatchMix] step={step} batch_total={total} unique={len(counts)} "
            + ", ".join(parts)
        )
