"""
Stage 2 policy Lightning callbacks.
"""

from __future__ import annotations

import math
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor

from common.batch_utils import select_primary_image_stream, temporal_frames_to_bcthw

logger = logging.getLogger(__name__)


@dataclass
class ThroughputLoggingConfig:
    enabled: bool = True
    log_every_n_steps: int = 10


def _default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _wrap_text(text: str, *, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        if sum(len(x) for x in current) + len(current) + len(w) > width and current:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def _render_panel(
    *,
    image: Image.Image,
    meta: str,
    instruction: str,
    gt: str,
    pred: str,
    freeform_pred: Optional[str] = None,
    width: int = 384,
) -> Image.Image:
    font = _default_font()
    image = image.convert("RGB").resize((width, width))

    text_lines = []
    if meta:
        text_lines += _wrap_text(meta, width=56) + [""]
    text_lines += ["instruction:"] + _wrap_text(instruction, width=56)
    text_lines += ["", "gt:"] + _wrap_text(gt, width=56)
    text_lines += ["", "pred:"] + _wrap_text(pred, width=56)
    if freeform_pred is not None:
        text_lines += ["", "freeform:"] + _wrap_text(str(freeform_pred), width=56)

    line_h = 14
    pad = 10
    text_h = pad * 2 + line_h * max(1, len(text_lines))
    panel = Image.new("RGB", (width, width + text_h), color=(255, 255, 255))
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    y = width + pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return panel


def _render_wide_panel(
    *,
    image: Image.Image,
    meta: str,
    instruction: str,
    gt: str,
    pred: str,
) -> Image.Image:
    font = _default_font()
    image = image.convert("RGB")

    wrap_w = max(56, min(200, image.width // 8))
    text_lines = []
    if meta:
        text_lines += _wrap_text(meta, width=wrap_w) + [""]
    text_lines += ["instruction:"] + _wrap_text(instruction, width=wrap_w)
    text_lines += ["", "gt:"] + _wrap_text(gt, width=wrap_w)
    text_lines += ["", "pred:"] + _wrap_text(pred, width=wrap_w)

    line_h = 14
    pad = 10
    text_h = pad * 2 + line_h * max(1, len(text_lines))
    panel = Image.new(
        "RGB", (image.width, image.height + text_h), color=(255, 255, 255)
    )
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    y = image.height + pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return panel


def _tensor_chw_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().cpu().to(torch.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.max().item() > 1.0 or x.min().item() < 0.0:
        x = x.clamp(0.0, 255.0) / 255.0
    x = x.clamp(0.0, 1.0)
    arr = (x * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(arr, mode="RGB")


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    draw.line([start, end], fill=color, width=3)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    mag = math.hypot(dx, dy)
    if mag <= 1.0e-12:
        return
    ux = dx / mag
    uy = dy / mag
    head_len = max(6.0, min(14.0, 0.22 * mag))
    spread = math.pi / 7.0

    left = (
        end[0] - head_len * (ux * math.cos(spread) + uy * math.sin(spread)),
        end[1] - head_len * (uy * math.cos(spread) - ux * math.sin(spread)),
    )
    right = (
        end[0] - head_len * (ux * math.cos(spread) - uy * math.sin(spread)),
        end[1] - head_len * (uy * math.cos(spread) + ux * math.sin(spread)),
    )
    draw.polygon([end, left, right], fill=color)


def _single_direction_panel(
    *,
    dx: float,
    dy: float,
    height: int,
    width: int,
    color: tuple[int, int, int],
) -> torch.Tensor:
    panel = Image.new("RGB", (int(width), int(height)), (18, 18, 18))
    draw = ImageDraw.Draw(panel)

    cx = width / 2.0
    cy = height / 2.0
    radius = 0.35 * min(width, height)

    mag = math.hypot(dx, dy)
    scale = 0.0 if mag <= 1.0e-12 else (radius / mag)

    draw.line([(0, cy), (width, cy)], fill=(60, 60, 60), width=1)
    draw.line([(cx, 0), (cx, height)], fill=(60, 60, 60), width=1)

    end = (cx + dx * scale, cy + dy * scale)
    _draw_arrow(draw, start=(cx, cy), end=end, color=color)
    return pil_to_tensor(panel).float() / 255.0


def _label_image(img: Image.Image, label: str) -> Image.Image:
    font = _default_font()
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([(0, 0), (out.width, 16)], fill=(255, 255, 255))
    draw.text((4, 2), label, fill=(0, 0, 0), font=font)
    return out


def _stitch_row(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("Expected at least one image")
    h = images[0].height
    total_w = sum(im.width for im in images)
    out = Image.new("RGB", (total_w, h), color=(255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _select_diverse_indices(
    *,
    episode_id: Any,
    frame_idx: Any,
    instructions: list[Any],
    max_items: int,
) -> list[int]:
    if max_items <= 0:
        return []

    n = min(len(instructions), max_items)
    if n <= 0:
        return []

    episode_ids: Optional[list[Any]] = (
        episode_id if isinstance(episode_id, list) else None
    )
    frame_idxs: Optional[list[Any]] = frame_idx if isinstance(frame_idx, list) else None

    def get_ep(i: int) -> str:
        if episode_ids is None or i >= len(episode_ids) or episode_ids[i] is None:
            return ""
        return str(episode_ids[i])

    def get_frame(i: int) -> str:
        if frame_idxs is None or i >= len(frame_idxs) or frame_idxs[i] is None:
            return ""
        return str(frame_idxs[i])

    chosen: list[int] = []

    # Prefer distinct (episode, instruction) pairs.
    seen_ep_instr: set[tuple[str, str]] = set()
    for i in range(len(instructions)):
        if len(chosen) >= n:
            break
        key = (get_ep(i), str(instructions[i]))
        if key in seen_ep_instr:
            continue
        seen_ep_instr.add(key)
        chosen.append(i)

    # If still short, prefer distinct (episode, frame_idx) within the same instruction.
    if len(chosen) < n:
        seen_ep_frame: set[tuple[str, str]] = set()
        for i in chosen:
            seen_ep_frame.add((get_ep(i), get_frame(i)))
        for i in range(len(instructions)):
            if len(chosen) >= n:
                break
            if i in chosen:
                continue
            key = (get_ep(i), get_frame(i))
            if key in seen_ep_frame:
                continue
            seen_ep_frame.add(key)
            chosen.append(i)

    # Fill remaining slots deterministically.
    if len(chosen) < n:
        for i in range(len(instructions)):
            if len(chosen) >= n:
                break
            if i not in chosen:
                chosen.append(i)

    return chosen[:n]


def _select_code_diverse_indices(*, gt_codes: list[Any], max_items: int) -> list[int]:
    """
    Prefer samples with distinct action-code sequences.

    This helps debugging when the first batch contains many near-duplicate
    instructions but different action codes.
    """
    if max_items <= 0:
        return []
    chosen: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for i, row in enumerate(gt_codes):
        if len(chosen) >= max_items:
            break
        if not isinstance(row, list):
            continue
        try:
            key = tuple(int(x) for x in row)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
    return chosen


def _safe_vector_mse(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) != len(b) or len(a) == 0:
        return None
    try:
        at = torch.tensor(a, dtype=torch.float32)
        bt = torch.tensor(b, dtype=torch.float32)
    except Exception:
        return None
    return float(torch.mean((at - bt) ** 2).item())


def _vector_summary(vec: Any, *, head_dims: int = 6) -> str:
    if not isinstance(vec, list):
        return str(vec)
    if len(vec) == 0:
        return "dim=0 []"
    try:
        vals = [float(x) for x in vec]
    except Exception:
        return str(vec)
    head = ", ".join(f"{x:.3f}" for x in vals[:head_dims])
    l2 = float(torch.tensor(vals, dtype=torch.float32).norm().item())
    return f"dim={len(vals)} l2={l2:.3f} head=[{head}]"


def _save_grid_and_records(
    *,
    panels: list[Image.Image],
    records: list[dict[str, Any]],
    out_dir: Path,
    prefix: str,
    step: int,
    trainer: pl.Trainer,
    wandb_key: str,
) -> None:
    if not panels:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = panels[0].size
    grid = Image.new("RGB", (w * len(panels), h), color=(255, 255, 255))
    for i, p in enumerate(panels):
        grid.paste(p, (i * w, 0))

    png_path = out_dir / f"{prefix}_step{step:06d}.png"
    json_path = out_dir / f"{prefix}_step{step:06d}.json"
    grid.save(png_path)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    try:
        import wandb  # type: ignore

        if hasattr(trainer, "logger") and getattr(trainer.logger, "experiment", None):
            # Optionally log a compact debug table when available (e.g., generated text
            # with padding/special tokens). This is useful for diagnosing decoding/parsing
            # issues without opening the JSON artifacts manually.
            try:
                cols = [
                    "instruction",
                    "gt_codes",
                    "pred_codes",
                    "prompt_padded_len",
                    "prompt_true_len",
                    "prompt_text_with_specials",
                    "generated_suffix_text_with_specials",
                ]
                rows = []
                for r in records:
                    dbg = r.get("gen_debug")
                    if not isinstance(dbg, dict):
                        continue
                    rows.append(
                        [
                            str(r.get("instruction", "")),
                            str(r.get("gt_codes")),
                            str(r.get("pred_codes")),
                            dbg.get("prompt_padded_len"),
                            dbg.get("prompt_true_len"),
                            dbg.get("prompt_text_with_specials"),
                            dbg.get("generated_suffix_text_with_specials"),
                        ]
                    )
                if rows:
                    trainer.logger.experiment.log(  # type: ignore[attr-defined]
                        {f"{wandb_key}_debug": wandb.Table(columns=cols, data=rows)}
                    )
            except Exception:
                pass
            trainer.logger.experiment.log(  # type: ignore[attr-defined]
                # Do not pass an explicit `step=` here: Lightning's WandB integration
                # may advance W&B's internal step counter differently (e.g., logging
                # multiple times per global_step), which causes W&B to drop these
                # image logs as "out of order".
                {wandb_key: wandb.Image(str(png_path))}
            )
    except Exception:
        pass


class ThroughputLoggingCallback(Callback):
    def __init__(self, cfg: Optional[ThroughputLoggingConfig] = None):
        super().__init__()
        self.cfg = cfg or ThroughputLoggingConfig()
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.cfg.enabled:
            return

        self._last_time = time.perf_counter()
        self._last_step = int(getattr(trainer, "global_step", 0))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self.cfg.enabled:
            return

        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if self.cfg.log_every_n_steps <= 0 or (step % self.cfg.log_every_n_steps) != 0:
            return

        now = time.perf_counter()
        if self._last_time is None or self._last_step is None:
            self._last_time = now
            self._last_step = step
            return

        dt = now - self._last_time
        ds = step - self._last_step
        if dt <= 0.0 or ds <= 0:
            self._last_time = now
            self._last_step = step
            return

        batch_size: Optional[int] = None
        try:
            frames = batch["frames"] if isinstance(batch, dict) else batch
            batch_size = int(getattr(frames, "shape", [None])[0])
        except Exception:
            batch_size = None

        steps_per_sec = float(ds) / float(dt)
        pl_module.log(
            "perf/steps_per_sec",
            steps_per_sec,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if batch_size is not None and batch_size > 0:
            pl_module.log(
                "perf/samples_per_sec",
                float(batch_size) * steps_per_sec,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        self._last_time = now
        self._last_step = step


@dataclass
class PolicyTrainSampleVizConfig:
    enabled: bool = True
    num_samples: int = 4
    every_n_steps: int = 500
    include_freeform_pred: bool = False
    freeform_max_new_tokens: int = 32


class PolicyTrainSampleVisualizationCallback(Callback):
    def __init__(self, cfg: Optional[PolicyTrainSampleVizConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyTrainSampleVizConfig()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self.cfg.enabled:
            return
        if not trainer.is_global_zero:
            return

        step = int(getattr(trainer, "global_step", 0))
        if self.cfg.every_n_steps <= 0 or (step % self.cfg.every_n_steps) != 0:
            return
        try:
            from stage2.backends.interfaces import BackendMode, Stage2Batch
        except Exception:
            logger.debug(
                "train sample viz: failed to import Stage2Batch", exc_info=True
            )
            return
        if not isinstance(batch, Stage2Batch):
            return

        action_tokens = getattr(pl_module, "action_tokens", None)
        backend = getattr(pl_module, "backend", None)
        if backend is None or not hasattr(backend, "latent_from_batch"):
            logger.debug(
                "train sample viz: missing pl_module.backend.latent_from_batch"
            )
            return

        logger.debug(
            "train sample viz: begin step=%d batch_idx=%d", step, int(batch_idx)
        )

        try:
            from stage2.backends.smolvla_shared.input_transform import (
                normalize_vector_mean_std,
            )
        except Exception:
            logger.debug("train sample viz: failed to import helpers", exc_info=True)
            return

        mode = getattr(pl_module, "backend_mode", BackendMode.CODES)
        if not isinstance(mode, BackendMode):
            try:
                mode = BackendMode(str(mode))
            except Exception:
                mode = BackendMode.CODES

        image_streams_all = batch.image_streams or {}
        if not image_streams_all:
            return
        frames = select_primary_image_stream(image_streams_all)
        if not isinstance(frames, torch.Tensor):
            return
        instructions = [str(x) for x in (batch.task_text or [])]
        if not instructions:
            return

        batch_meta = batch.meta if isinstance(batch.meta, dict) else {}
        episode_id = batch_meta.get("episode_id")
        episode_id_list = episode_id if isinstance(episode_id, list) else None
        frame_idx = batch_meta.get("frame_idx")
        frame_idx_list = frame_idx if isinstance(frame_idx, list) else None
        dataset_name = batch_meta.get("dataset_name")
        dataset_name_list = dataset_name if isinstance(dataset_name, list) else None

        # Select diverse samples from this batch.
        max_items = min(self.cfg.num_samples, len(instructions))
        if max_items <= 0:
            return

        chosen = _select_diverse_indices(
            episode_id=episode_id_list,
            frame_idx=frame_idx_list,
            instructions=list(instructions),
            max_items=max_items,
        )

        try:
            frames_sel = frames[chosen]
        except Exception:
            logger.debug("train sample viz: failed to slice frames", exc_info=True)
            return
        instr_sel = [str(instructions[i]) for i in chosen]

        # Recompute images for the selected frames to avoid processing the whole batch.
        try:
            images_sel = pl_module.frames_to_images(frames_sel)
        except Exception:
            logger.debug(
                "train sample viz: frames_to_images failed on selection", exc_info=True
            )
            return

        # Compute GT targets for the selected subset.
        gt_codes_sel: list[list[int]] | None = None
        gt_vectors_sel: list[list[float]] | None = None
        gt_actions_sel: list[list[float]] | None = None
        try:
            video_sel = temporal_frames_to_bcthw(frames_sel)
            if video_sel.dtype == torch.uint8:
                video_sel = video_sel.float().div(255.0)
            else:
                video_sel = video_sel.float()
            if mode is BackendMode.CODES:
                gt_codes_t = pl_module.code_provider.codes_from_video(video_sel)
                gt_codes_sel = [row.tolist() for row in gt_codes_t.detach().cpu()]
            elif mode is BackendMode.LATENT_FLOW:
                gt_codes_t, gt_vectors_t = (
                    pl_module.code_provider.codes_and_vectors_from_video(video_sel)
                )
                gt_codes_sel = [row.tolist() for row in gt_codes_t.detach().cpu()]
                gt_vectors_sel = (
                    gt_vectors_t.detach()
                    .cpu()
                    .reshape(gt_vectors_t.shape[0], -1)
                    .tolist()
                )
            elif mode is BackendMode.ACTIONS:
                gt_actions_t = batch.target_actions
                if gt_actions_t is None:
                    return
                gt_actions_sel = gt_actions_t[chosen].detach().cpu().tolist()
            elif mode is BackendMode.MULTITASK:
                gt_codes_t, gt_vectors_t = (
                    pl_module.code_provider.codes_and_vectors_from_video(video_sel)
                )
                gt_codes_sel = [row.tolist() for row in gt_codes_t.detach().cpu()]
                gt_vectors_sel = (
                    gt_vectors_t.detach()
                    .cpu()
                    .reshape(gt_vectors_t.shape[0], -1)
                    .tolist()
                )
                gt_actions_t = batch.target_actions
                if gt_actions_t is None:
                    return
                gt_actions_sel = gt_actions_t[chosen].detach().cpu().tolist()
            else:
                return
        except Exception:
            logger.debug("train sample viz: GT target extraction failed", exc_info=True)
            return

        pred_codes_sel: list[list[int]] | None = None
        pred_vectors_sel: list[list[float]] | None = None
        pred_actions_sel: list[list[float]] | None = None
        gen_debug: Optional[list[dict[str, Any]]] = None
        try:
            image_streams = pl_module._extract_policy_image_streams(frames_sel)
            state = batch.state
            if state is not None:
                state = state[chosen]
                state = normalize_vector_mean_std(
                    value=state,
                    stats=getattr(pl_module, "normalization_stats", None),
                    key_candidates=["observation.state", "initial_state", "state"],
                )
            latent = backend.latent_from_batch(
                Stage2Batch(
                    image_streams=image_streams,
                    image_padding_masks=pl_module._extract_policy_image_padding_masks(
                        image_streams
                    ),
                    task_text=instr_sel,
                    state=state,
                ),
                mode=mode,
            )
            tokens = latent.tokens
            if tokens is None and isinstance(latent.logits, torch.Tensor):
                tokens = latent.logits.argmax(dim=-1)
            if isinstance(tokens, torch.Tensor):
                pred_codes_sel = tokens.detach().cpu().tolist()
            if isinstance(latent.vector, torch.Tensor):
                pred_vectors_sel = (
                    latent.vector.detach()
                    .cpu()
                    .reshape(latent.vector.shape[0], -1)
                    .tolist()
                )
            if isinstance(latent.actions, torch.Tensor):
                pred_actions_sel = latent.actions.detach().cpu().tolist()
            meta = latent.meta if isinstance(latent.meta, dict) else {}
            gen_debug = (
                meta.get("parse_debug")
                if isinstance(meta.get("parse_debug"), list)
                else None
            )
        except Exception:
            logger.debug("train sample viz: backend prediction failed", exc_info=True)
            return

        if (
            pred_codes_sel is None
            and pred_vectors_sel is None
            and pred_actions_sel is None
        ):
            logger.debug("train sample viz: backend returned no prediction outputs")
            return

        freeform_texts: Optional[list[str]] = None
        if self.cfg.include_freeform_pred and hasattr(
            pl_module, "_predict_freeform_text"
        ):
            policy_model = getattr(pl_module, "policy_model", None)
            if policy_model is None or not hasattr(policy_model, "generate"):
                freeform_texts = None
            else:
                try:
                    freeform_texts = pl_module._predict_freeform_text(  # type: ignore[attr-defined]
                        frames=frames_sel,
                        instructions=instr_sel,
                        max_new_tokens=int(self.cfg.freeform_max_new_tokens),
                    )
                except Exception:
                    logger.debug(
                        "train sample viz: freeform generation failed", exc_info=True
                    )
                    freeform_texts = None

        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"

        panels: list[Image.Image] = []
        records: list[dict[str, Any]] = []
        for rank, original_idx in enumerate(chosen):
            gt_str = ""
            pred_str = ""
            if (
                mode is BackendMode.CODES
                and action_tokens is not None
                and gt_codes_sel is not None
                and pred_codes_sel is not None
            ):
                gt_str = action_tokens.format_target(gt_codes_sel[rank])
                try:
                    pred_str = action_tokens.format_target(pred_codes_sel[rank])
                except Exception:
                    pred_str = f"<INVALID> {pred_codes_sel[rank]}"
            elif (
                mode is BackendMode.LATENT_FLOW
                and gt_vectors_sel is not None
                and pred_vectors_sel is not None
            ):
                gt_str = f"latent_gt: {_vector_summary(gt_vectors_sel[rank])}"
                pred_str = f"latent_pred: {_vector_summary(pred_vectors_sel[rank])}"
                mse = _safe_vector_mse(gt_vectors_sel[rank], pred_vectors_sel[rank])
                if mse is not None:
                    pred_str = f"{pred_str} mse={mse:.4f}"
            elif (
                mode is BackendMode.ACTIONS
                and gt_actions_sel is not None
                and pred_actions_sel is not None
            ):
                gt_str = f"action_gt: {_vector_summary(gt_actions_sel[rank])}"
                pred_str = f"action_pred: {_vector_summary(pred_actions_sel[rank])}"
                mse = _safe_vector_mse(gt_actions_sel[rank], pred_actions_sel[rank])
                if mse is not None:
                    pred_str = f"{pred_str} mse={mse:.4f}"
            elif mode is BackendMode.MULTITASK:
                latent_gt = (
                    _vector_summary(gt_vectors_sel[rank])
                    if gt_vectors_sel is not None
                    else "n/a"
                )
                latent_pred = (
                    _vector_summary(pred_vectors_sel[rank])
                    if pred_vectors_sel is not None
                    else "n/a"
                )
                action_gt = (
                    _vector_summary(gt_actions_sel[rank])
                    if gt_actions_sel is not None
                    else "n/a"
                )
                action_pred = (
                    _vector_summary(pred_actions_sel[rank])
                    if pred_actions_sel is not None
                    else "n/a"
                )
                gt_str = f"latent_gt: {latent_gt} action_gt: {action_gt}"
                pred_str = f"latent_pred: {latent_pred} action_pred: {action_pred}"

                latent_mse = (
                    _safe_vector_mse(gt_vectors_sel[rank], pred_vectors_sel[rank])
                    if gt_vectors_sel is not None and pred_vectors_sel is not None
                    else None
                )
                action_mse = (
                    _safe_vector_mse(gt_actions_sel[rank], pred_actions_sel[rank])
                    if gt_actions_sel is not None and pred_actions_sel is not None
                    else None
                )
                suffix = []
                if latent_mse is not None:
                    suffix.append(f"latent_mse={latent_mse:.4f}")
                if action_mse is not None:
                    suffix.append(f"action_mse={action_mse:.4f}")
                if suffix:
                    pred_str = f"{pred_str} {' '.join(suffix)}"
            freeform = (
                freeform_texts[rank]
                if freeform_texts is not None and rank < len(freeform_texts)
                else None
            )
            meta_parts: list[str] = []
            if (
                episode_id_list is not None
                and frame_idx_list is not None
                and original_idx < len(episode_id_list)
                and original_idx < len(frame_idx_list)
            ):
                meta_parts.append(
                    f"episode_id: {episode_id_list[original_idx]}  frame_idx: {frame_idx_list[original_idx]}"
                )
            if dataset_name_list is not None and original_idx < len(dataset_name_list):
                meta_parts.append(f"dataset: {dataset_name_list[original_idx]}")

            panels.append(
                _render_panel(
                    image=images_sel[rank],
                    meta="  ".join(meta_parts),
                    instruction=str(instructions[original_idx]),
                    gt=gt_str,
                    pred=pred_str,
                    freeform_pred=freeform,
                )
            )
            records.append(
                {
                    "step": step,
                    "mode": mode.value,
                    "index": int(original_idx),
                    "rank": int(rank),
                    "instruction": str(instructions[original_idx]),
                    "gt_codes": (
                        gt_codes_sel[rank] if gt_codes_sel is not None else None
                    ),
                    "pred_codes": (
                        pred_codes_sel[rank] if pred_codes_sel is not None else None
                    ),
                    "gt_vector": (
                        gt_vectors_sel[rank] if gt_vectors_sel is not None else None
                    ),
                    "pred_vector": (
                        pred_vectors_sel[rank] if pred_vectors_sel is not None else None
                    ),
                    "gt_action": (
                        gt_actions_sel[rank] if gt_actions_sel is not None else None
                    ),
                    "pred_action": (
                        pred_actions_sel[rank] if pred_actions_sel is not None else None
                    ),
                    "pred_freeform": freeform,
                    "gen_debug": (
                        gen_debug[rank]
                        if isinstance(gen_debug, list) and rank < len(gen_debug)
                        else None
                    ),
                    "episode_id": (
                        episode_id_list[original_idx]
                        if episode_id_list and original_idx < len(episode_id_list)
                        else None
                    ),
                    "frame_idx": (
                        frame_idx_list[original_idx]
                        if frame_idx_list and original_idx < len(frame_idx_list)
                        else None
                    ),
                    "dataset_name": (
                        dataset_name_list[original_idx]
                        if dataset_name_list and original_idx < len(dataset_name_list)
                        else None
                    ),
                }
            )

        _save_grid_and_records(
            panels=panels,
            records=records,
            out_dir=out_dir,
            prefix="train_samples",
            step=step,
            trainer=trainer,
            wandb_key="train/samples",
        )
