"""
Common Lightning callbacks shared across stages.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Optional

import torch
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from common.batch_utils import select_primary_image_stream, temporal_frames_to_bcthw


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


def _find_image_logger(trainer: Any) -> Any | None:
    loggers = getattr(trainer, "loggers", None)
    if loggers is None:
        logger = getattr(trainer, "logger", None)
        loggers = [] if logger is None else [logger]
    for logger in loggers:
        if logger is None:
            continue
        if callable(getattr(logger, "log_image", None)):
            return logger
    return None


def _metadata_value(values: Any, idx: int) -> Any:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if idx >= len(values):
            return None
        return values[idx]
    if isinstance(values, torch.Tensor):
        if values.ndim == 0:
            return values.item()
        if idx >= int(values.shape[0]):
            return None
        item = values[idx]
        if isinstance(item, torch.Tensor) and item.ndim == 0:
            return item.item()
        return item
    return values


def _normalize_image_tensor(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu().to(torch.float32)
    if x.max().item() > 1.0 or x.min().item() < 0.0:
        x = x.clamp(0.0, 255.0) / 255.0
    return x.clamp(0.0, 1.0)


@dataclass
class DataSampleVisualizationConfig:
    enabled: bool = True
    every_n_steps: int = 500
    num_samples: int = 8
    key: str = "data/train_samples"
    mode: str = "stage1"  # stage1 -> [frame_t, frame_t+1], stage2 -> frame_t only


class DataSampleVisualizationCallback(Callback):
    """
    Lightweight batch visualization callback with no model forward pass.

    Stage1 mode:
    - logs [frame_t, frame_t+1] pairs.

    Stage2 mode:
    - logs frame_t only, with task metadata caption.
    """

    def __init__(self, cfg: Optional[DataSampleVisualizationConfig] = None):
        super().__init__()
        self.cfg = cfg or DataSampleVisualizationConfig()

    @staticmethod
    def _extract_video_and_meta(
        batch: Any,
    ) -> tuple[torch.Tensor | None, dict[str, Any], list[str]]:
        if isinstance(batch, dict):
            frames = batch.get("frames")
            if not isinstance(frames, torch.Tensor):
                return None, {}, []
            try:
                video = temporal_frames_to_bcthw(frames)
            except ValueError:
                return None, {}, []
            meta = batch.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            task_text = batch.get("task_text")
            if isinstance(task_text, list):
                tasks = [str(x) for x in task_text]
            else:
                tasks = []
            return video, meta, tasks

        image_streams = getattr(batch, "image_streams", None)
        if not isinstance(image_streams, dict) or not image_streams:
            return None, {}, []
        frames = select_primary_image_stream(image_streams)
        if not isinstance(frames, torch.Tensor):
            return None, {}, []
        video = temporal_frames_to_bcthw(frames)
        meta = getattr(batch, "meta", None)
        if not isinstance(meta, dict):
            meta = {}
        task_text = getattr(batch, "task_text", None)
        if isinstance(task_text, (list, tuple)):
            tasks = [str(x) for x in task_text]
        else:
            tasks = []
        return video, meta, tasks

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch_idx
        if not bool(self.cfg.enabled):
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if (
            int(self.cfg.every_n_steps) <= 0
            or (step % int(self.cfg.every_n_steps)) != 0
        ):
            return

        image_logger = _find_image_logger(trainer)
        if image_logger is None:
            return

        video, meta, task_text = self._extract_video_and_meta(batch)
        if video is None or int(video.shape[0]) <= 0:
            return

        batch_size = int(video.shape[0])
        n = min(int(self.cfg.num_samples), batch_size)
        if n <= 0:
            return

        mode = str(self.cfg.mode)
        video = _normalize_image_tensor(video[:n])
        if mode == "stage1":
            if int(video.shape[2]) < 2:
                return
            frame_t = video[:, :, 0]
            frame_t_plus = video[:, :, 1]
            stacked = torch.stack([frame_t, frame_t_plus], dim=1).reshape(
                n * 2, *frame_t.shape[1:]
            )
            grid = make_grid(stacked, nrow=2, normalize=False)
        else:
            frame_t = video[:, :, 0]
            grid = make_grid(frame_t, nrow=min(n, 4), normalize=False)

        captions: list[str] = []
        dataset_short = meta.get("dataset_short") if isinstance(meta, dict) else None
        episode_id = meta.get("episode_id") if isinstance(meta, dict) else None
        frame_idx = meta.get("frame_idx") if isinstance(meta, dict) else None
        for i in range(n):
            ds = _metadata_value(dataset_short, i)
            ep = _metadata_value(episode_id, i)
            fi = _metadata_value(frame_idx, i)
            task = task_text[i] if i < len(task_text) else ""
            parts = []
            if ds is not None:
                parts.append(f"dataset={ds}")
            if ep is not None and fi is not None:
                parts.append(f"episode={ep} frame={fi}")
            if task:
                parts.append(f"task={task}")
            captions.append(" | ".join(parts))

        try:
            image_logger.log_image(
                key=str(self.cfg.key),
                images=[grid],
                caption=["\n".join(captions)] if captions else [f"step={step}"],
            )
        except Exception:
            return


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

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch: Any, batch_idx: int
    ) -> None:
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
            if (
                self.log_batch_composition_every_n_steps > 0
                and (step + 1) % self.log_batch_composition_every_n_steps == 0
            ):
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
        msg = (
            f"[{prefix}][DatasetUsage] step={step} interval_total={total} "
            + ", ".join(parts)
        )
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
