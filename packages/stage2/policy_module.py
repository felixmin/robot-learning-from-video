"""
Stage 2 policy LightningModule driven by a policy backend.

Flow:
1) Take a Stage2Batch containing image streams + text + optional state/action targets.
2) Run frozen LAM (Stage 1) to produce target codes [B, S].
3) Delegate prompting/masking + LM loss + constrained generation/parsing to the backend.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from common.batch_utils import (
    move_dataclass_tensors_to_device,
    select_primary_image_stream,
    temporal_frames_to_bcthw,
    uint8_image_streams_to_float32,
)
from stage2.backends.interfaces import BackendMode, Stage2Batch, PolicyBackend
from stage2.backends.smolvla_shared.input_transform import normalize_vector_mean_std
from stage2.constrained_decode import ActionTokenIds
from stage2.online_lam import (
    LatentCodeProvider,
)


@dataclass
class PolicyOptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01


class PolicyLightningModule(pl.LightningModule):
    """
    Stage 2 module: image + language -> LAM code tokens.

    Notes:
    - `code_provider` is frozen and used only to compute supervision.
    - The backend owns model-specific details (processor, chat templating, masking, decoding/parsing).
    """

    def __init__(
        self,
        *,
        backend: PolicyBackend,
        code_provider: LatentCodeProvider | None,
        backend_mode: BackendMode = BackendMode.CODES,
        normalization_stats: dict[str, dict[str, Any]] | None = None,
        optimizer: PolicyOptimizerConfig | None = None,
        action_token_ids: ActionTokenIds | None = None,
        train_teacher_forced_metrics_every_n_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend  # should be an nn.Module so Lightning can optimize it
        self.code_provider = code_provider
        self.backend_mode = backend_mode
        self.normalization_stats = normalization_stats
        self.optimizer_cfg = optimizer or PolicyOptimizerConfig()
        self.action_token_ids = action_token_ids
        self.train_teacher_forced_metrics_every_n_steps = (
            train_teacher_forced_metrics_every_n_steps
        )

        self._val_batch_payload_queue: deque[dict[str, Any]] = deque()

    def _require_code_provider(self) -> LatentCodeProvider:
        if self.code_provider is None:
            raise ValueError(
                f"`code_provider` is required for backend_mode={self.backend_mode.value!r}."
            )
        return self.code_provider

    @property
    def policy_model(self) -> Any:
        model = getattr(self.backend, "policy_model", None)
        if model is not None:
            return model
        core = getattr(self.backend, "core", None)
        return getattr(core, "vlm", None)

    @property
    def processor(self) -> Any:
        proc = getattr(self.backend, "processor", None)
        if proc is not None:
            return proc
        core = getattr(self.backend, "core", None)
        return getattr(core, "processor", None)

    @property
    def action_tokens(self) -> Any:
        cfg = getattr(self.backend, "cfg", None)
        return getattr(cfg, "action_tokens", None)

    @property
    def chat(self) -> Any:
        cfg = getattr(self.backend, "cfg", None)
        return getattr(cfg, "chat", None)

    @property
    def frames_to_images(self) -> Any:
        adapter = getattr(self.backend, "frames_to_images", None)
        if adapter is not None:
            return adapter
        core = getattr(self.backend, "core", None)
        return getattr(core, "frames_to_images", None)

    @staticmethod
    def _sanitize_metric_suffix(name: str) -> str:
        out = []
        for ch in name:
            if ch.isalnum() or ch in "_":
                out.append(ch.lower())
            else:
                out.append("_")
        suffix = "".join(out).strip("_")
        return suffix or "unknown"

    @staticmethod
    def _extract_policy_image_streams(frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(
                f"Expected temporal frames tensor [B,T,...], got shape {tuple(frames.shape)}"
            )
        if frames.shape[-1] == 3:
            return {"observation.images.rgb": frames[:, 0, ...]}
        if frames.shape[2] == 3:
            return {"observation.images.rgb": frames[:, 0, ...]}
        if frames.shape[1] == 3:
            return {"observation.images.rgb": frames[:, :, 0, ...]}
        raise ValueError(
            "Unrecognized frames layout; expected last dim=3 (BTHWC), shape[2]=3 (BTCHW), "
            f"or shape[1]=3 (BCTHW). Got {tuple(frames.shape)}"
        )

    @staticmethod
    def _extract_policy_image_padding_masks(
        image_streams: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for key, stream in image_streams.items():
            masks[key] = torch.ones(
                (int(stream.shape[0]),), dtype=torch.bool, device=stream.device
            )
        return masks

    def _log_vector_stats(
        self, *, prefix: str, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, float]:
        if pred.shape != gt.shape or pred.numel() == 0:
            return {}
        diff = pred - gt
        pred_flat = pred.reshape(pred.shape[0], -1)
        gt_flat = gt.reshape(gt.shape[0], -1)
        diff_flat = diff.reshape(diff.shape[0], -1)
        eps = 1.0e-8

        pred_norm = torch.linalg.norm(pred_flat, dim=1)
        gt_norm = torch.linalg.norm(gt_flat, dim=1)
        cos_sim = torch.sum(pred_flat * gt_flat, dim=1) / (pred_norm * gt_norm + eps)
        pred_q = torch.round(pred_flat * 1.0e3).to(torch.int32)
        gt_q = torch.round(gt_flat * 1.0e3).to(torch.int32)
        pred_unique = torch.unique(pred_q, dim=0).shape[0]
        gt_unique = torch.unique(gt_q, dim=0).shape[0]

        stats = {
            "pred_mean": float(pred_flat.mean().item()),
            "gt_mean": float(gt_flat.mean().item()),
            "pred_std": float(pred_flat.std(unbiased=False).item()),
            "gt_std": float(gt_flat.std(unbiased=False).item()),
            "pred_l2_mean": float(pred_norm.mean().item()),
            "gt_l2_mean": float(gt_norm.mean().item()),
            "mse": float(torch.mean(diff_flat**2).item()),
            "mae": float(torch.mean(torch.abs(diff_flat)).item()),
            "cosine_mean": float(cos_sim.mean().item()),
            "pred_unique": float(pred_unique),
            "gt_unique": float(gt_unique),
            "pred_unique_frac": float(pred_unique) / float(pred_flat.shape[0]),
            "gt_unique_frac": float(gt_unique) / float(gt_flat.shape[0]),
        }
        for key, value in stats.items():
            self.log(f"{prefix}/{key}", value, prog_bar=False, sync_dist=True)
        return stats

    @staticmethod
    def _metadata_value_at(values: Any, idx: int) -> Any:
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
            if isinstance(item, torch.Tensor):
                if item.ndim == 0:
                    return item.item()
                return item.detach().cpu().tolist()
            return item
        return values

    def _build_val_batch_payload(
        self,
        *,
        batch: Stage2Batch,
        frames: torch.Tensor,
        instructions: list[str],
        codes: torch.Tensor | None,
        vectors: torch.Tensor | None,
        actions: torch.Tensor | None,
        pred_tokens: torch.Tensor | None,
        pred_vectors: torch.Tensor | None,
        pred_actions: torch.Tensor | None,
        gen_debug: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        max_items = min(64, len(instructions), int(frames.shape[0]))
        if codes is not None:
            max_items = min(max_items, int(codes.shape[0]))
        if vectors is not None:
            max_items = min(max_items, int(vectors.shape[0]))
        if actions is not None:
            max_items = min(max_items, int(actions.shape[0]))
        if pred_tokens is not None:
            max_items = min(max_items, int(pred_tokens.shape[0]))
        if pred_vectors is not None:
            max_items = min(max_items, int(pred_vectors.shape[0]))
        if pred_actions is not None:
            max_items = min(max_items, int(pred_actions.shape[0]))

        batch_meta = batch.meta if isinstance(batch.meta, dict) else {}
        episode_id = batch_meta.get("episode_id")
        frame_idx = batch_meta.get("frame_idx")
        dataset_name = batch_meta.get("dataset_name")
        dataset_short = batch_meta.get("dataset_short")
        language = batch_meta.get("language")
        task = batch_meta.get("task")

        records: list[dict[str, Any]] = []
        for i in range(max_items):
            rec = {
                "index": int(i),
                "mode": self.backend_mode.value,
                "frame": frames[i].detach().cpu(),
                "instruction": str(instructions[i]),
                "gt_codes": (
                    codes[i].detach().cpu().tolist() if codes is not None else None
                ),
                "pred_codes": (
                    pred_tokens[i].to(torch.long).detach().cpu().tolist()
                    if pred_tokens is not None
                    else None
                ),
                "gt_vector": (
                    vectors[i].detach().cpu().reshape(-1).tolist()
                    if vectors is not None
                    else None
                ),
                "pred_vector": (
                    pred_vectors[i].detach().cpu().reshape(-1).tolist()
                    if pred_vectors is not None
                    else None
                ),
                "gt_action": (
                    actions[i].detach().cpu().tolist() if actions is not None else None
                ),
                "pred_action": (
                    pred_actions[i].detach().cpu().tolist()
                    if pred_actions is not None
                    else None
                ),
                "gen_debug": (
                    gen_debug[i]
                    if isinstance(gen_debug, list) and i < len(gen_debug)
                    else None
                ),
                "metadata": {
                    "episode_id": self._metadata_value_at(episode_id, i),
                    "frame_idx": self._metadata_value_at(frame_idx, i),
                    "dataset_name": self._metadata_value_at(dataset_name, i),
                    "dataset_short": self._metadata_value_at(dataset_short, i),
                    "language": self._metadata_value_at(language, i),
                    "task": self._metadata_value_at(task, i),
                },
            }
            records.append(rec)

        return {"records": records}

    def consume_next_val_batch_payload(self) -> dict[str, Any] | None:
        if not self._val_batch_payload_queue:
            return None
        return self._val_batch_payload_queue.popleft()

    def reset_val_batch_payload_queue(self) -> None:
        self._val_batch_payload_queue.clear()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, _codes, _vectors, _actions, _frames, _instructions = (
            self._loss_and_targets_from_batch(batch)
        )
        self.log("train/loss", out.loss, prog_bar=True, sync_dist=True)
        for key, value in out.metrics.items():
            if key == "loss":
                continue
            self.log(f"train/{key}", float(value), prog_bar=False, sync_dist=True)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, codes, vectors, actions, frames, instructions = (
            self._loss_and_targets_from_batch(batch)
        )
        self.log("val/loss", out.loss, prog_bar=True, sync_dist=True)
        for key, value in out.metrics.items():
            if key == "loss":
                continue
            self.log(f"val/{key}", float(value), prog_bar=False, sync_dist=True)

        if not isinstance(batch, Stage2Batch):
            raise TypeError("Stage 2 validation expects Stage2Batch.")

        image_streams = self._extract_policy_image_streams(frames)
        state = batch.state
        if state is None:
            raise ValueError("Stage2Batch must include state for validation.")
        state = normalize_vector_mean_std(
            value=state,
            stats=self.normalization_stats,
            key_candidates=["observation.state", "initial_state", "state"],
        )
        latent = self.backend.latent_from_batch(
            Stage2Batch(
                image_streams=image_streams,
                image_padding_masks=self._extract_policy_image_padding_masks(
                    image_streams
                ),
                task_text=instructions,
                state=state,
            ),
            mode=self.backend_mode,
        )
        pred = latent.tokens
        pred_vector = latent.vector
        pred_actions = latent.actions

        vector_stats: dict[str, float] | None = None
        action_stats: dict[str, float] | None = None
        dataset_mix: dict[str, int] | None = None

        meta = latent.meta if isinstance(latent.meta, dict) else {}
        gen_debug = (
            meta.get("parse_debug")
            if isinstance(meta.get("parse_debug"), list)
            else None
        )

        if batch_idx == 0:
            if pred is not None and codes is not None:
                gt = codes.to(device=pred.device, dtype=torch.long)
                pred = pred.to(torch.long)
                if pred.shape == gt.shape and pred.numel() > 0:
                    matches = (pred == gt).to(torch.float32)
                    self.log(
                        "val/token_accuracy",
                        matches.mean().to(self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    self.log(
                        "val/sequence_accuracy",
                        matches.all(dim=1).to(torch.float32).mean().to(self.device),
                        prog_bar=False,
                        sync_dist=True,
                    )

            if pred_vector is not None and vectors is not None:
                gt_vec = vectors.reshape(vectors.shape[0], -1).to(
                    device=pred_vector.device, dtype=pred_vector.dtype
                )
                if pred_vector.shape == gt_vec.shape and pred_vector.numel() > 0:
                    vec_mse = torch.mean((pred_vector - gt_vec) ** 2)
                    self.log(
                        "val/latent_vector_mse",
                        vec_mse.to(self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    vector_stats = self._log_vector_stats(
                        prefix="val/latent_vector_stats", pred=pred_vector, gt=gt_vec
                    )

            if pred_actions is not None and actions is not None:
                gt_actions = actions.to(
                    device=pred_actions.device, dtype=pred_actions.dtype
                )
                if pred_actions.shape == gt_actions.shape and pred_actions.numel() > 0:
                    act_mse = torch.mean((pred_actions - gt_actions) ** 2)
                    self.log(
                        "val/action_mse",
                        act_mse.to(self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    action_stats = self._log_vector_stats(
                        prefix="val/action_stats", pred=pred_actions, gt=gt_actions
                    )

            if gen_debug:
                start_frac = torch.tensor(
                    sum(
                        1
                        for r in gen_debug
                        if isinstance(r, dict) and r.get("has_action_start")
                    )
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                end_frac = torch.tensor(
                    sum(
                        1
                        for r in gen_debug
                        if isinstance(r, dict) and r.get("has_action_end")
                    )
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                mean_codes = torch.tensor(
                    sum(
                        int(r.get("num_codes_parsed", 0))
                        for r in gen_debug
                        if isinstance(r, dict)
                    )
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.log(
                    "val/gen_has_action_start_frac",
                    start_frac,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "val/gen_has_action_end_frac",
                    end_frac,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "val/gen_num_codes_parsed_mean",
                    mean_codes,
                    prog_bar=False,
                    sync_dist=True,
                )

            batch_meta = batch.meta if isinstance(batch.meta, dict) else None
            dataset_names = (
                batch_meta.get("dataset_short") if batch_meta is not None else None
            )
            if isinstance(dataset_names, list) and dataset_names:
                counts = Counter(
                    str(x) if x is not None else "None" for x in dataset_names
                )
                total = float(len(dataset_names))
                top = counts.most_common(8)
                dataset_mix = {name: int(count) for name, count in top}
                self.log(
                    "val/dataset_mix_unique",
                    float(len(counts)),
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "val/dataset_mix_top1_frac",
                    float(top[0][1]) / total,
                    prog_bar=False,
                    sync_dist=True,
                )
                probs = torch.tensor(
                    [float(count) / total for _, count in counts.items()],
                    device=self.device,
                    dtype=torch.float32,
                )
                if probs.numel() > 1:
                    entropy = -torch.sum(probs * torch.log(probs + 1.0e-8))
                    max_entropy = torch.log(
                        torch.tensor(float(probs.numel()), device=self.device)
                    )
                    entropy_norm = entropy / (max_entropy + 1.0e-8)
                    self.log(
                        "val/dataset_mix_entropy_norm",
                        entropy_norm,
                        prog_bar=False,
                        sync_dist=True,
                    )
                for name, count in top:
                    suffix = self._sanitize_metric_suffix(name)
                    self.log(
                        f"val/dataset_mix_frac_{suffix}",
                        float(count) / total,
                        prog_bar=False,
                        sync_dist=True,
                    )
                details = ", ".join(f"{name}={count}" for name, count in top)
                print(
                    f"[Val][BatchMix] step={int(getattr(self.trainer, 'global_step', 0))} "
                    f"batch_total={int(total)} unique={len(counts)} {details}"
                )

        try:
            payload = self._build_val_batch_payload(
                batch=batch,
                frames=frames,
                instructions=instructions,
                codes=codes,
                vectors=vectors,
                actions=actions,
                pred_tokens=pred,
                pred_vectors=pred_vector,
                pred_actions=pred_actions,
                gen_debug=gen_debug,
            )
            self._val_batch_payload_queue.append(payload)
        except Exception:
            pass

        return out.loss

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, Stage2Batch):
            batch = move_dataclass_tensors_to_device(batch, device)
            if batch.image_streams is not None:
                batch = Stage2Batch(
                    image_streams=uint8_image_streams_to_float32(batch.image_streams),
                    image_padding_masks=batch.image_padding_masks,
                    task_text=batch.task_text,
                    subtask_text=batch.subtask_text,
                    language_tokens=batch.language_tokens,
                    language_attention_mask=batch.language_attention_mask,
                    target_codes=batch.target_codes,
                    target_latent_vectors=batch.target_latent_vectors,
                    target_actions=batch.target_actions,
                    action_is_pad=batch.action_is_pad,
                    state=batch.state,
                    meta=batch.meta,
                )
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @staticmethod
    def _resize_lam_video(
        video: torch.Tensor, target_image_size: tuple[int, int]
    ) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected LAM video [B,C,T,H,W], got {tuple(video.shape)}"
            )
        target_hw = tuple(int(x) for x in target_image_size)
        if tuple(int(x) for x in video.shape[3:]) == target_hw:
            return video
        b, c, t, h, w = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
        return (
            flat.reshape(b, t, c, target_hw[0], target_hw[1])
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @classmethod
    def _lam_video_from_stage2_batch(
        cls,
        batch: Stage2Batch,
        *,
        target_image_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = select_primary_image_stream(batch.image_streams or {})
        video = temporal_frames_to_bcthw(frames, expected_time_steps=2)
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 255.0
        else:
            video = video.to(torch.float32)
        if target_image_size is not None:
            video = cls._resize_lam_video(video, target_image_size)
        return video, frames

    def _loss_and_targets_from_stage2_batch(
        self,
        batch: Stage2Batch,
    ) -> tuple[
        Any,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        list[str],
    ]:
        target_image_size = None
        if self.code_provider is not None:
            raw_image_size = getattr(self.code_provider, "image_size", None)
            if raw_image_size is not None:
                target_image_size = tuple(int(x) for x in raw_image_size)
        video, frames = self._lam_video_from_stage2_batch(
            batch,
            target_image_size=target_image_size,
        )
        instructions = [str(x) for x in (batch.task_text or [])]
        actions: torch.Tensor | None = None
        action_is_pad: torch.Tensor | None = None
        codes: torch.Tensor | None = None
        vectors: torch.Tensor | None = None
        state = batch.state
        if state is None:
            raise ValueError(
                "Stage2Batch must include state for current Stage 2 training."
            )

        if self.backend_mode is BackendMode.CODES:
            code_provider = self._require_code_provider()
            codes = code_provider.codes_from_video(video).to(torch.long).detach().cpu()
        elif self.backend_mode is BackendMode.LATENT_FLOW:
            code_provider = self._require_code_provider()
            codes, vectors = code_provider.codes_and_vectors_from_video(video)
            codes = codes.to(torch.long).detach().cpu()
            vectors = vectors.detach().cpu()
        elif self.backend_mode is BackendMode.MULTITASK:
            code_provider = self._require_code_provider()
            codes, vectors = code_provider.codes_and_vectors_from_video(video)
            codes = codes.to(torch.long).detach().cpu()
            vectors = vectors.detach().cpu()
            actions = (
                None
                if batch.target_actions is None
                else batch.target_actions.detach().cpu()
            )
            action_is_pad = batch.action_is_pad
        elif self.backend_mode is BackendMode.ACTIONS:
            actions = (
                None
                if batch.target_actions is None
                else batch.target_actions.detach().cpu()
            )
            action_is_pad = batch.action_is_pad
        else:
            raise NotImplementedError(f"Unsupported backend mode: {self.backend_mode}")

        state = normalize_vector_mean_std(
            value=state,
            stats=self.normalization_stats,
            key_candidates=["observation.state", "initial_state", "state"],
        )
        if actions is not None:
            actions = normalize_vector_mean_std(
                value=actions,
                stats=self.normalization_stats,
                key_candidates=["action", "ACTION"],
            )

        out = self.backend.loss_from_batch(
            Stage2Batch(
                image_streams=batch.image_streams,
                image_padding_masks=batch.image_padding_masks,
                task_text=instructions,
                target_codes=codes,
                target_latent_vectors=vectors,
                target_actions=actions,
                action_is_pad=action_is_pad,
                state=state,
                meta=batch.meta,
            ),
            mode=self.backend_mode,
        )
        return out, codes, vectors, actions, frames, instructions

    def _loss_and_targets_from_batch(self, batch: Any) -> tuple[
        Any,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        list[str],
    ]:
        if not isinstance(batch, Stage2Batch):
            raise TypeError("Stage 2 training expects Stage2Batch.")
        return self._loss_and_targets_from_stage2_batch(batch)

    @torch.no_grad()
    def _predict_freeform_text(
        self,
        *,
        frames: torch.Tensor,
        instructions: list[str],
        max_new_tokens: int = 32,
    ) -> list[str]:
        model = self.policy_model
        processor = self.processor
        chat = self.chat
        frames_to_images = self.frames_to_images
        if (
            model is None
            or processor is None
            or chat is None
            or frames_to_images is None
        ):
            raise RuntimeError(
                "Backend must expose policy_model, processor, chat, and frames_to_images for freeform."
            )

        from stage2.policy_inputs import build_prompt_inputs

        images = frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=processor,
            images=images,
            instructions=instructions,
            chat=chat,
            device=self.device,
        )

        input_ids = prompt_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("processor output input_ids must be a 2D tensor")
        prompt_len = int(input_ids.shape[1])

        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        tok = getattr(processor, "tokenizer", None)
        decode = getattr(tok, "decode", None) if tok is not None else None
        if decode is None:
            raise TypeError("processor.tokenizer must implement decode(...)")

        texts: list[str] = []
        for i in range(int(generated.shape[0])):
            suffix_ids = generated[i, prompt_len:].tolist()
            texts.append(str(decode(suffix_ids, skip_special_tokens=False)))
        return texts

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
