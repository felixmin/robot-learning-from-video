from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from PIL import Image
from torchvision.utils import flow_to_image

from common.batch_utils import temporal_frames_to_bcthw
from foundation.callbacks import (
    _label_image,
    _render_panel,
    _render_wide_panel,
    _safe_vector_mse,
    _save_grid_and_records,
    _select_diverse_indices,
    _single_direction_panel,
    _stitch_row,
    _tensor_chw_to_pil,
    _vector_summary,
)
from foundation.validation_cache import Stage2ValidationCache


class Stage2ValidationCheck:
    def __init__(
        self,
        *,
        name: str,
        enabled: bool = True,
        min_samples: int = 1,
        buckets: Optional[list[str]] = None,
        **_: Any,
    ) -> None:
        self.name = str(name)
        self.enabled = bool(enabled)
        self.min_samples = int(min_samples)
        self.buckets = [str(x) for x in (buckets or [])]

    def required_metadata(self) -> list[str]:
        return []

    @staticmethod
    def no_output(reason: str) -> dict[str, Any]:
        return {"_produced": 0, "_reason": str(reason)}

    @staticmethod
    def success(*, produced: int = 1, metrics: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        out = dict(metrics or {})
        out["_produced"] = max(0, int(produced))
        if out["_produced"] > 0:
            out.pop("_reason", None)
        return out

    def can_run(self, cache: Stage2ValidationCache) -> tuple[bool, str]:
        if cache.sample_count() < self.min_samples:
            return False, f"Only {cache.sample_count()} samples (need {self.min_samples})"
        required = self.required_metadata()
        if not required:
            return True, ""

        count = 0
        for rec in cache.get_records():
            meta = rec.get("metadata")
            if not isinstance(meta, dict):
                continue
            if all(meta.get(key) is not None for key in required):
                count += 1

        if count < self.min_samples:
            return False, f"Only {count} samples with {required} (need {self.min_samples})"
        return True, ""

    def run(
        self,
        cache: Stage2ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError


class SamplePanelsCheck(Stage2ValidationCheck):
    def __init__(
        self,
        *,
        name: str,
        enabled: bool = True,
        min_samples: int = 1,
        buckets: Optional[list[str]] = None,
        num_samples: int = 4,
        include_freeform_pred: bool = False,
        freeform_max_new_tokens: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            enabled=enabled,
            min_samples=min_samples,
            buckets=buckets,
            **kwargs,
        )
        self.num_samples = int(num_samples)
        self.include_freeform_pred = bool(include_freeform_pred)
        self.freeform_max_new_tokens = int(freeform_max_new_tokens)

    def _choose_records(self, cache: Stage2ValidationCache) -> list[tuple[int, dict[str, Any]]]:
        records = cache.get_records()
        if not records:
            return []

        n = min(self.num_samples, len(records))
        if n <= 0:
            return []

        if cache.fixed_indices and cache.fixed_records:
            selected: list[tuple[int, dict[str, Any]]] = []
            fixed_set = set(cache.fixed_indices)
            for idx, rec in zip(cache.fixed_indices, cache.fixed_records):
                selected.append((int(idx), rec))
                if len(selected) >= n:
                    return selected

            remaining = n - len(selected)
            pool = [i for i in range(len(records)) if i not in fixed_set]
            if remaining > 0 and pool:
                take = torch.randperm(len(pool))[:remaining].tolist()
                for j in take:
                    i = pool[j]
                    selected.append((i, records[i]))
            return selected

        episode_id = [
            rec.get("metadata", {}).get("episode_id") if isinstance(rec.get("metadata"), dict) else None
            for rec in records
        ]
        frame_idx = [
            rec.get("metadata", {}).get("frame_idx") if isinstance(rec.get("metadata"), dict) else None
            for rec in records
        ]
        instructions = [str(rec.get("instruction", "")) for rec in records]
        chosen = _select_diverse_indices(
            episode_id=episode_id,
            frame_idx=frame_idx,
            instructions=instructions,
            max_items=n,
        )
        return [(i, records[i]) for i in chosen]

    @staticmethod
    def _normalize_images(raw: Any) -> list[Image.Image]:
        images: list[Image.Image] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Image.Image):
                    images.append(item)
                elif isinstance(item, torch.Tensor) and item.ndim == 3:
                    images.append(_tensor_chw_to_pil(item))
            return images

        if isinstance(raw, torch.Tensor):
            if raw.ndim == 4:
                for i in range(int(raw.shape[0])):
                    images.append(_tensor_chw_to_pil(raw[i]))
                return images
            if raw.ndim == 5:
                video = temporal_frames_to_bcthw(raw)
                first = video[:, :, 0]
                for i in range(int(first.shape[0])):
                    images.append(_tensor_chw_to_pil(first[i]))
                return images

        return images

    @classmethod
    def _frames_to_images_with_fallback(
        cls,
        *,
        pl_module: pl.LightningModule,
        frames_batched: torch.Tensor,
    ) -> list[Image.Image]:
        adapter = getattr(pl_module, "frames_to_images", None)
        if callable(adapter):
            images = cls._normalize_images(adapter(frames_batched))
            if images:
                return images

        video = temporal_frames_to_bcthw(frames_batched)
        first = video[:, :, 0]
        return [_tensor_chw_to_pil(first[i]) for i in range(int(first.shape[0]))]

    def run(
        self,
        cache: Stage2ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, Any]:
        if not trainer.is_global_zero:
            return self.no_output("not_global_zero")

        chosen = self._choose_records(cache)
        if not chosen:
            return self.no_output("no_selected_records")

        frames: list[torch.Tensor] = []
        valid_chosen: list[tuple[int, dict[str, Any]]] = []
        for idx, rec in chosen:
            frame = rec.get("frame")
            if isinstance(frame, torch.Tensor):
                frames.append(frame)
                valid_chosen.append((idx, rec))

        if not frames:
            return self.no_output("selected_records_missing_frame_tensors")

        try:
            images = self._frames_to_images_with_fallback(
                pl_module=pl_module,
                frames_batched=torch.stack(frames, dim=0),
            )
        except Exception as e:
            raise RuntimeError(f"{self.name}: failed to convert frames to images") from e
        if not images:
            raise RuntimeError(f"{self.name}: no images produced for rendering")

        n_items = min(len(valid_chosen), len(images))
        chosen = valid_chosen[:n_items]
        images = images[:n_items]
        if n_items <= 0:
            return self.no_output("no_valid_items_after_image_conversion")

        freeform_texts: Optional[list[str]] = None
        if self.include_freeform_pred and hasattr(pl_module, "_predict_freeform_text"):
            vla_model = getattr(pl_module, "vla_model", None)
            if vla_model is not None and hasattr(vla_model, "generate"):
                try:
                    instructions = [str(rec.get("instruction", "")) for _, rec in chosen]
                    out = pl_module._predict_freeform_text(  # type: ignore[attr-defined]
                        frames=torch.stack([rec.get("frame") for _, rec in chosen], dim=0),
                        instructions=instructions,
                        max_new_tokens=int(self.freeform_max_new_tokens),
                    )
                    if isinstance(out, list):
                        freeform_texts = [str(x) for x in out]
                except Exception:
                    freeform_texts = None

        action_tokens = getattr(pl_module, "action_tokens", None)
        step = int(getattr(trainer, "global_step", 0))
        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"

        panels: list[Image.Image] = []
        records_out: list[dict[str, Any]] = []
        for rank, ((idx, rec), image) in enumerate(zip(chosen, images)):
            mode = str(rec.get("mode") or "")
            gt_codes = rec.get("gt_codes")
            pred_codes = rec.get("pred_codes")
            gt_vector = rec.get("gt_vector")
            pred_vector = rec.get("pred_vector")
            gt_action = rec.get("gt_action")
            pred_action = rec.get("pred_action")

            if mode == "codes" and action_tokens is not None and isinstance(gt_codes, list):
                gt_str = action_tokens.format_target(gt_codes)
                if isinstance(pred_codes, list):
                    try:
                        pred_str = action_tokens.format_target(pred_codes)
                    except Exception:
                        pred_str = f"<INVALID> {pred_codes}"
                else:
                    pred_str = "<MISSING>"
            elif mode == "latent_flow" and isinstance(gt_vector, list):
                gt_str = f"latent_gt: {_vector_summary(gt_vector)}"
                if isinstance(pred_vector, list):
                    pred_str = f"latent_pred: {_vector_summary(pred_vector)}"
                    mse = _safe_vector_mse(gt_vector, pred_vector)
                    if mse is not None:
                        pred_str = f"{pred_str} mse={mse:.4f}"
                else:
                    pred_str = "latent_pred: <MISSING>"
            elif mode == "actions" and isinstance(gt_action, list):
                gt_str = f"action_gt: {_vector_summary(gt_action)}"
                if isinstance(pred_action, list):
                    pred_str = f"action_pred: {_vector_summary(pred_action)}"
                    mse = _safe_vector_mse(gt_action, pred_action)
                    if mse is not None:
                        pred_str = f"{pred_str} mse={mse:.4f}"
                else:
                    pred_str = "action_pred: <MISSING>"
            elif mode == "multitask":
                latent_gt = _vector_summary(gt_vector) if isinstance(gt_vector, list) else "n/a"
                latent_pred = _vector_summary(pred_vector) if isinstance(pred_vector, list) else "n/a"
                action_gt = _vector_summary(gt_action) if isinstance(gt_action, list) else "n/a"
                action_pred = _vector_summary(pred_action) if isinstance(pred_action, list) else "n/a"
                gt_str = f"latent_gt: {latent_gt} action_gt: {action_gt}"
                pred_str = f"latent_pred: {latent_pred} action_pred: {action_pred}"
                suffix: list[str] = []
                latent_mse = _safe_vector_mse(gt_vector, pred_vector)
                action_mse = _safe_vector_mse(gt_action, pred_action)
                if latent_mse is not None:
                    suffix.append(f"latent_mse={latent_mse:.4f}")
                if action_mse is not None:
                    suffix.append(f"action_mse={action_mse:.4f}")
                if suffix:
                    pred_str = f"{pred_str} {' '.join(suffix)}"
            else:
                gt_str = str(gt_codes if gt_codes is not None else "")
                pred_str = str(pred_codes if pred_codes is not None else "")

            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            meta_parts: list[str] = []
            ep = meta.get("episode_id")
            fi = meta.get("frame_idx")
            ds = meta.get("dataset_name")
            if ep is not None and fi is not None:
                meta_parts.append(f"episode_id: {ep}  frame_idx: {fi}")
            if ds is not None:
                meta_parts.append(f"dataset: {ds}")

            freeform = freeform_texts[rank] if freeform_texts is not None and rank < len(freeform_texts) else None
            panels.append(
                _render_panel(
                    image=image,
                    meta="  ".join(meta_parts),
                    instruction=str(rec.get("instruction", "")),
                    gt=gt_str,
                    pred=pred_str,
                    freeform_pred=freeform,
                )
            )
            records_out.append(
                {
                    "step": step,
                    "mode": mode,
                    "index": int(idx),
                    "rank": int(rank),
                    "instruction": str(rec.get("instruction", "")),
                    "gt_codes": gt_codes,
                    "pred_codes": pred_codes,
                    "gt_vector": gt_vector,
                    "pred_vector": pred_vector,
                    "gt_action": gt_action,
                    "pred_action": pred_action,
                    "pred_freeform": freeform,
                    "gen_debug": rec.get("gen_debug"),
                    "episode_id": ep,
                    "frame_idx": fi,
                    "dataset_name": ds,
                }
            )

        prefix = "val_samples" if not metric_suffix else f"val_samples{metric_suffix}"
        wandb_key = "val/samples" if not metric_suffix else f"val/samples{metric_suffix}"
        _save_grid_and_records(
            panels=panels,
            records=records_out,
            out_dir=out_dir,
            prefix=prefix,
            step=step,
            trainer=trainer,
            wandb_key=wandb_key,
        )
        return self.success(produced=int(n_items))


class LatentFlowDecodeCheck(Stage2ValidationCheck):
    def __init__(
        self,
        *,
        name: str,
        laq_checkpoint_path: str,
        enabled: bool = True,
        min_samples: int = 1,
        buckets: Optional[list[str]] = None,
        num_samples: int = 4,
        max_decode_batch_size: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            enabled=enabled,
            min_samples=min_samples,
            buckets=buckets,
            **kwargs,
        )
        self.laq_checkpoint_path = str(laq_checkpoint_path or "")
        self.num_samples = int(num_samples)
        self.max_decode_batch_size = int(max_decode_batch_size)
        self._laq_model: Optional[torch.nn.Module] = None
        self._laq_device: Optional[torch.device] = None
        self._load_failed = False

    def _ensure_laq_model(self, device: torch.device) -> Optional[torch.nn.Module]:
        if self._load_failed:
            return None
        if self._laq_model is None:
            from laq.checkpoints import load_laq_task_from_checkpoint

            try:
                task = load_laq_task_from_checkpoint(self.laq_checkpoint_path, map_location="cpu", strict=True)
            except Exception:
                self._load_failed = True
                return None
            model = task.model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._laq_model = model
            self._laq_device = torch.device("cpu")

        if self._laq_model is None:
            return None
        if self._laq_device != device:
            self._laq_model = self._laq_model.to(device)
            self._laq_device = device
        return self._laq_model

    @staticmethod
    def _vectors_to_actions(vectors_flat: torch.Tensor, *, laq_model: torch.nn.Module) -> torch.Tensor:
        bsz = int(vectors_flat.shape[0])
        code_seq_len = int(laq_model.code_seq_len)
        codebook_dim = int(laq_model.vq.codebooks.shape[1])
        expected = code_seq_len * codebook_dim
        if vectors_flat.shape[1] != expected:
            raise ValueError(f"Unexpected latent vector dim={vectors_flat.shape[1]} (expected {expected})")
        vec = vectors_flat.reshape(bsz, code_seq_len, codebook_dim)
        act = laq_model.vq.project_out(vec)
        action_h, action_w = laq_model.action_shape
        return act.reshape(bsz, 1, int(action_h), int(action_w), int(act.shape[-1]))

    def run(
        self,
        cache: Stage2ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, Any]:
        if not trainer.is_global_zero:
            return self.no_output("not_global_zero")

        applicable: list[dict[str, Any]] = []
        for rec in cache.get_records():
            mode = str(rec.get("mode") or "")
            if mode not in {"latent_flow", "multitask"}:
                continue
            if not isinstance(rec.get("frame"), torch.Tensor):
                continue
            if not isinstance(rec.get("gt_vector"), list):
                continue
            if not isinstance(rec.get("pred_vector"), list):
                continue
            applicable.append(rec)

        if not applicable:
            return self.no_output("no_applicable_records")

        laq_model = self._ensure_laq_model(pl_module.device)
        if laq_model is None:
            return self.no_output("laq_model_unavailable")
        if getattr(laq_model, "flow_decoder", None) is None:
            return self.no_output("missing_flow_decoder")
        if getattr(laq_model, "decoder_context_projection", None) is None:
            return self.no_output("missing_decoder_context_projection")

        num = min(self.num_samples, len(applicable))
        if num <= 0:
            return self.no_output("num_samples_zero")

        episode_id = [
            rec.get("metadata", {}).get("episode_id") if isinstance(rec.get("metadata"), dict) else None
            for rec in applicable
        ]
        frame_idx = [
            rec.get("metadata", {}).get("frame_idx") if isinstance(rec.get("metadata"), dict) else None
            for rec in applicable
        ]
        instructions = [str(rec.get("instruction", "")) for rec in applicable]
        selected = _select_diverse_indices(
            episode_id=episode_id,
            frame_idx=frame_idx,
            instructions=instructions,
            max_items=num,
        )
        if not selected:
            return self.no_output("no_selected_indices")

        step = int(getattr(trainer, "global_step", 0))
        out_dir = Path(str(trainer.default_root_dir)) / "visualizations"
        panels: list[Image.Image] = []
        records_out: list[dict[str, Any]] = []

        max_chunk = max(1, self.max_decode_batch_size)
        for start in range(0, len(selected), max_chunk):
            chunk_idx = selected[start : start + max_chunk]
            frames_chunk = torch.stack([applicable[i]["frame"] for i in chunk_idx], dim=0)
            frames_chunk = temporal_frames_to_bcthw(frames_chunk.to(pl_module.device), expected_time_steps=2)

            try:
                gt_chunk = torch.tensor(
                    [applicable[i]["gt_vector"] for i in chunk_idx],
                    dtype=torch.float32,
                    device=pl_module.device,
                )
                pred_chunk = torch.tensor(
                    [applicable[i]["pred_vector"] for i in chunk_idx],
                    dtype=torch.float32,
                    device=pl_module.device,
                )
            except Exception:
                continue

            from laq.models.flow import compute_weighted_mean_flow

            first_frame = frames_chunk[:, :, :1]
            rest_frame = frames_chunk[:, :, 1:]
            first_frame_f = first_frame.to(torch.float32)
            rest_frame_f = rest_frame.to(torch.float32)
            if first_frame.dtype == torch.uint8:
                first_frame_f = first_frame_f / 255.0
            if rest_frame.dtype == torch.uint8:
                rest_frame_f = rest_frame_f / 255.0

            pixel_context = laq_model.decoder_context_projection(first_frame_f)
            h_dec, w_dec = laq_model.patch_height_width
            attn_bias = laq_model.spatial_rel_pos_bias(h_dec, w_dec, device=pl_module.device)
            gt_actions = self._vectors_to_actions(gt_chunk, laq_model=laq_model)
            pred_actions = self._vectors_to_actions(pred_chunk, laq_model=laq_model)

            gt_flow = laq_model.flow_decoder(pixel_context, gt_actions, attn_bias)
            pred_flow = laq_model.flow_decoder(pixel_context, pred_actions, attn_bias)
            static_eps = float(getattr(laq_model.flow_config, "summary_static_eps", 1e-6))
            gt_dx, gt_dy = compute_weighted_mean_flow(gt_flow, static_eps=static_eps)
            pred_dx, pred_dy = compute_weighted_mean_flow(pred_flow, static_eps=static_eps)

            gt_flow_rgb = flow_to_image(gt_flow.detach().cpu()).float() / 255.0
            pred_flow_rgb = flow_to_image(pred_flow.detach().cpu()).float() / 255.0
            first_cpu = first_frame_f.detach().cpu()
            rest_cpu = rest_frame_f.detach().cpu()
            gt_dx = gt_dx.detach().cpu()
            gt_dy = gt_dy.detach().cpu()
            pred_dx = pred_dx.detach().cpu()
            pred_dy = pred_dy.detach().cpu()

            for local_i, record_idx in enumerate(chunk_idx):
                rec = applicable[record_idx]
                frame_t_img = _tensor_chw_to_pil(first_cpu[local_i, :, 0])
                frame_h_img = _tensor_chw_to_pil(rest_cpu[local_i, :, 0])
                gt_flow_img = _tensor_chw_to_pil(gt_flow_rgb[local_i])
                pred_flow_img = _tensor_chw_to_pil(pred_flow_rgb[local_i])
                h, w = int(frame_t_img.height), int(frame_t_img.width)
                gt_dir = _single_direction_panel(
                    dx=float(gt_dx[local_i].item()),
                    dy=float(gt_dy[local_i].item()),
                    height=h,
                    width=w,
                    color=(80, 220, 80),
                )
                pred_dir = _single_direction_panel(
                    dx=float(pred_dx[local_i].item()),
                    dy=float(pred_dy[local_i].item()),
                    height=h,
                    width=w,
                    color=(230, 90, 90),
                )
                gt_dir_img = _tensor_chw_to_pil(gt_dir)
                pred_dir_img = _tensor_chw_to_pil(pred_dir)

                row = _stitch_row(
                    [
                        _label_image(frame_t_img, "frame_t"),
                        _label_image(frame_h_img, "frame_t+k"),
                        _label_image(gt_flow_img, "flow_gt(latent)"),
                        _label_image(pred_flow_img, "flow_pred(latent)"),
                        _label_image(gt_dir_img, "dir_gt"),
                        _label_image(pred_dir_img, "dir_pred"),
                    ]
                )
                meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                meta_parts: list[str] = []
                ep = meta.get("episode_id")
                fi = meta.get("frame_idx")
                ds = meta.get("dataset_name")
                if ep is not None and fi is not None:
                    meta_parts.append(f"episode_id: {ep} frame_idx: {fi}")
                if ds is not None:
                    meta_parts.append(f"dataset: {ds}")
                panel = _render_wide_panel(
                    image=row,
                    meta="  ".join(meta_parts),
                    instruction=str(rec.get("instruction", "")),
                    gt=f"latent_gt: {_vector_summary(rec.get('gt_vector'))}",
                    pred=f"latent_pred: {_vector_summary(rec.get('pred_vector'))}",
                )
                panels.append(panel)
                records_out.append(
                    {
                        "step": step,
                        "mode": rec.get("mode"),
                        "index": int(record_idx),
                        "instruction": str(rec.get("instruction", "")),
                        "episode_id": ep,
                        "frame_idx": fi,
                        "dataset_name": ds,
                        "gt_mean_flow": [float(gt_dx[local_i].item()), float(gt_dy[local_i].item())],
                        "pred_mean_flow": [float(pred_dx[local_i].item()), float(pred_dy[local_i].item())],
                        "latent_vector_mse": _safe_vector_mse(rec.get("gt_vector"), rec.get("pred_vector")),
                    }
                )

        if not panels:
            return self.no_output("no_panels_rendered")

        prefix = "val_latent_flow_decode" if not metric_suffix else f"val_latent_flow_decode{metric_suffix}"
        wandb_key = "val/latent_flow_decode" if not metric_suffix else f"val/latent_flow_decode{metric_suffix}"
        _save_grid_and_records(
            panels=panels,
            records=records_out,
            out_dir=out_dir,
            prefix=prefix,
            step=step,
            trainer=trainer,
            wandb_key=wandb_key,
        )
        return self.success(produced=int(len(panels)))


class TokenDistributionCheck(Stage2ValidationCheck):
    @staticmethod
    def _entropy(counts: torch.Tensor) -> float:
        probs = counts.to(torch.float32)
        denom = float(probs.sum().item())
        if denom <= 0.0:
            return 0.0
        probs = probs / denom
        nonzero = probs[probs > 0]
        if nonzero.numel() == 0:
            return 0.0
        return float((-nonzero * torch.log(nonzero)).sum().item())

    def run(
        self,
        cache: Stage2ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        *,
        metric_suffix: str = "",
    ) -> dict[str, Any]:
        del trainer

        all_codes = cache.get_all_gt_codes()
        if all_codes is None or all_codes.numel() == 0:
            return self.no_output("no_gt_codes")

        codes_flat = all_codes.flatten().to(torch.long)
        action_tokens = getattr(pl_module, "action_tokens", None)
        codebook_size = int(getattr(action_tokens, "codebook_size", 0)) if action_tokens is not None else 0
        if codebook_size <= 0:
            codebook_size = int(torch.max(codes_flat).item()) + 1

        counts = torch.bincount(codes_flat, minlength=max(1, codebook_size))
        used = int((counts > 0).sum().item())

        sequences = [tuple(row.tolist()) for row in all_codes]
        unique_seq_count = len(set(sequences))

        metrics = {
            f"val/token_distribution_entropy{metric_suffix}": self._entropy(counts),
            f"val/token_distribution_utilization{metric_suffix}": float(used) / float(max(1, codebook_size)),
            f"val/token_distribution_unique_sequences{metric_suffix}": float(unique_seq_count),
        }
        pl_module.log_dict(metrics, sync_dist=True)
        return self.success(produced=int(all_codes.shape[0]), metrics=metrics)


CHECK_TYPES = {
    "sample_panels": SamplePanelsCheck,
    "latent_flow_decode": LatentFlowDecodeCheck,
    "token_distribution": TokenDistributionCheck,
}
