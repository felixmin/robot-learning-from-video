"""
Stage 2 (Foundation) LightningModule driven by a VLA backend.

Flow:
1) Take an OXE/OpenX batch containing frames + language.
2) Run frozen LAQ (Stage 1) to produce target codes [B, S].
3) Delegate prompting/masking + LM loss + constrained generation/parsing to the backend.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch

from foundation.backends.interfaces import BackendMode, FoundationBatch, VLABackend
from foundation.backends.smolvla_shared.input_transform import normalize_vector_mean_std
from foundation.constrained_decode import ActionTokenIds
from foundation.online_laq import (
    LatentCodeProvider,
    extract_oxe_actions,
    extract_oxe_initial_state,
    extract_oxe_language,
    oxe_frames_to_laq_video,
)


@dataclass
class VLAOptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01


class VLATokenBackendLightningModule(pl.LightningModule):
    """
    Stage 2 module: image + language -> LAQ code tokens.

    Notes:
    - `code_provider` is frozen and used only to compute supervision.
    - The backend owns model-specific details (processor, chat templating, masking, decoding/parsing).
    """

    def __init__(
        self,
        *,
        backend: VLABackend,
        code_provider: LatentCodeProvider,
        backend_mode: BackendMode = BackendMode.CODES,
        normalization_stats: dict[str, dict[str, Any]] | None = None,
        optimizer: VLAOptimizerConfig | None = None,
        action_token_ids: ActionTokenIds | None = None,
        train_teacher_forced_metrics_every_n_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend  # should be an nn.Module so Lightning can optimize it
        self.code_provider = code_provider
        self.backend_mode = backend_mode
        self.normalization_stats = normalization_stats
        self.optimizer_cfg = optimizer or VLAOptimizerConfig()
        self.action_token_ids = action_token_ids
        self.train_teacher_forced_metrics_every_n_steps = train_teacher_forced_metrics_every_n_steps

        # Stashed for visualization callback (rank0 only reads it).
        self._last_val_sample: dict[str, Any] | None = None

    @property
    def vla_model(self) -> Any:
        model = getattr(self.backend, "vla_model", None)
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
            raise ValueError(f"Expected OXE frames tensor [B,T,...], got shape {tuple(frames.shape)}")
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
            masks[key] = torch.ones((int(stream.shape[0]),), dtype=torch.bool, device=stream.device)
        return masks

    def _log_vector_stats(self, *, prefix: str, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, _codes, _vectors, _actions, _frames, _instructions = self._loss_and_targets_from_oxe_batch(batch)
        self.log("train/loss", out.loss, prog_bar=True, sync_dist=True)
        for key, value in out.metrics.items():
            if key == "loss":
                continue
            self.log(f"train/{key}", float(value), prog_bar=False, sync_dist=True)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, codes, vectors, actions, frames, instructions = self._loss_and_targets_from_oxe_batch(batch)
        self.log("val/loss", out.loss, prog_bar=True, sync_dist=True)
        for key, value in out.metrics.items():
            if key == "loss":
                continue
            self.log(f"val/{key}", float(value), prog_bar=False, sync_dist=True)

        if batch_idx == 0:
            image_streams = self._extract_policy_image_streams(frames)
            state = extract_oxe_initial_state({"initial_state": batch["initial_state"]})
            state = normalize_vector_mean_std(
                value=state,
                stats=self.normalization_stats,
                key_candidates=["observation.state", "initial_state", "state"],
            )
            latent = self.backend.latent_from_batch(
                FoundationBatch(
                    image_streams=image_streams,
                    image_padding_masks=self._extract_policy_image_padding_masks(image_streams),
                    task_text=instructions,
                    state=state,
                ),
                mode=self.backend_mode,
            )
            pred = latent.tokens
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

            pred_vector = latent.vector
            vector_stats: dict[str, float] | None = None
            if pred_vector is not None and vectors is not None:
                gt_vec = vectors.reshape(vectors.shape[0], -1).to(device=pred_vector.device, dtype=pred_vector.dtype)
                if pred_vector.shape == gt_vec.shape and pred_vector.numel() > 0:
                    vec_mse = torch.mean((pred_vector - gt_vec) ** 2)
                    self.log("val/latent_vector_mse", vec_mse.to(self.device), prog_bar=True, sync_dist=True)
                    vector_stats = self._log_vector_stats(prefix="val/latent_vector_stats", pred=pred_vector, gt=gt_vec)

            pred_actions = latent.actions
            action_stats: dict[str, float] | None = None
            if pred_actions is not None and actions is not None:
                gt_actions = actions.to(device=pred_actions.device, dtype=pred_actions.dtype)
                if pred_actions.shape == gt_actions.shape and pred_actions.numel() > 0:
                    act_mse = torch.mean((pred_actions - gt_actions) ** 2)
                    self.log("val/action_mse", act_mse.to(self.device), prog_bar=True, sync_dist=True)
                    action_stats = self._log_vector_stats(
                        prefix="val/action_stats", pred=pred_actions, gt=gt_actions
                    )

            meta = latent.meta if isinstance(latent.meta, dict) else {}
            gen_debug = meta.get("parse_debug") if isinstance(meta.get("parse_debug"), list) else None
            if gen_debug:
                start_frac = torch.tensor(
                    sum(1 for r in gen_debug if isinstance(r, dict) and r.get("has_action_start"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                end_frac = torch.tensor(
                    sum(1 for r in gen_debug if isinstance(r, dict) and r.get("has_action_end"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                mean_codes = torch.tensor(
                    sum(int(r.get("num_codes_parsed", 0)) for r in gen_debug if isinstance(r, dict))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.log("val/gen_has_action_start_frac", start_frac, prog_bar=False, sync_dist=True)
                self.log("val/gen_has_action_end_frac", end_frac, prog_bar=False, sync_dist=True)
                self.log("val/gen_num_codes_parsed_mean", mean_codes, prog_bar=False, sync_dist=True)

            dataset_mix: dict[str, int] | None = None
            dataset_names = batch.get("dataset_name") if isinstance(batch, dict) else None
            if isinstance(dataset_names, list) and dataset_names:
                counts = Counter(str(x) if x is not None else "None" for x in dataset_names)
                total = float(len(dataset_names))
                top = counts.most_common(8)
                dataset_mix = {name: int(count) for name, count in top}
                self.log("val/dataset_mix_unique", float(len(counts)), prog_bar=False, sync_dist=True)
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
                    max_entropy = torch.log(torch.tensor(float(probs.numel()), device=self.device))
                    entropy_norm = entropy / (max_entropy + 1.0e-8)
                    self.log("val/dataset_mix_entropy_norm", entropy_norm, prog_bar=False, sync_dist=True)
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

            # Save a small sample for visualization callbacks.
            try:
                max_items = min(64, len(instructions))
                if codes is not None:
                    max_items = min(max_items, int(codes.shape[0]))
                if vectors is not None:
                    max_items = min(max_items, int(vectors.shape[0]))
                if actions is not None:
                    max_items = min(max_items, int(actions.shape[0]))
                episode_id = batch.get("episode_id") if isinstance(batch, dict) else None
                frame_idx = batch.get("frame_idx") if isinstance(batch, dict) else None
                dataset_name = batch.get("dataset_name") if isinstance(batch, dict) else None
                pred_list: list[list[int]] | None = None
                if isinstance(pred, torch.Tensor):
                    pred_list = pred[:max_items].detach().cpu().tolist()
                pred_vector_list: list[list[float]] | None = None
                if isinstance(pred_vector, torch.Tensor):
                    pred_vector_list = (
                        pred_vector[:max_items].detach().cpu().reshape(max_items, -1).tolist()
                    )
                gt_vector_list: list[list[float]] | None = None
                if isinstance(vectors, torch.Tensor):
                    gt_vector_list = vectors[:max_items].detach().cpu().reshape(max_items, -1).tolist()
                pred_action_list: list[list[float]] | None = None
                if isinstance(pred_actions, torch.Tensor):
                    pred_action_list = pred_actions[:max_items].detach().cpu().tolist()
                gt_action_list: list[list[float]] | None = None
                if isinstance(actions, torch.Tensor):
                    gt_action_list = actions[:max_items].detach().cpu().tolist()
                self._last_val_sample = {
                    "frames": frames[:max_items].detach().cpu(),
                    "instructions": list(instructions[:max_items]),
                    "mode": self.backend_mode.value,
                    "gt_codes": [row.tolist() for row in codes[:max_items].detach().cpu()] if codes is not None else None,
                    "pred_codes": [list(row) for row in pred_list] if pred_list is not None else None,
                    "gt_vectors": gt_vector_list,
                    "pred_vectors": pred_vector_list,
                    "gt_actions": gt_action_list,
                    "pred_actions": pred_action_list,
                    "gen_debug": gen_debug[:max_items] if isinstance(gen_debug, list) else None,
                    "episode_id": list(episode_id[:max_items]) if episode_id is not None else None,
                    "frame_idx": list(frame_idx[:max_items]) if frame_idx is not None else None,
                    "dataset_name": list(dataset_name[:max_items]) if dataset_name is not None else None,
                    "vector_stats": vector_stats,
                    "action_stats": action_stats,
                    "dataset_mix": dataset_mix,
                }
            except Exception:
                self._last_val_sample = None

        return out.loss

    def _loss_and_targets_from_oxe_batch(
        self, batch: Any
    ) -> tuple[Any, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor, list[str]]:
        if not isinstance(batch, dict):
            raise TypeError("Expected OXE batch dict with keys including 'frames' and 'language'.")

        frames = batch["frames"]
        instructions = extract_oxe_language(batch)
        video = oxe_frames_to_laq_video(frames)
        actions: torch.Tensor | None = None
        action_is_pad: torch.Tensor | None = None
        codes: torch.Tensor | None = None
        vectors: torch.Tensor | None = None
        state = extract_oxe_initial_state({"initial_state": batch["initial_state"]})

        if self.backend_mode is BackendMode.CODES:
            codes = self.code_provider.codes_from_video(video).to(torch.long).detach().cpu()
        elif self.backend_mode is BackendMode.LATENT_FLOW:
            codes, vectors = self.code_provider.codes_and_vectors_from_video(video)
            codes = codes.to(torch.long).detach().cpu()
            vectors = vectors.detach().cpu()
        elif self.backend_mode is BackendMode.MULTITASK:
            codes, vectors = self.code_provider.codes_and_vectors_from_video(video)
            codes = codes.to(torch.long).detach().cpu()
            vectors = vectors.detach().cpu()
            actions = extract_oxe_actions(batch).detach().cpu()
            action_is_pad = batch["action_is_pad"]
        elif self.backend_mode is BackendMode.ACTIONS:
            actions = extract_oxe_actions(batch).detach().cpu()
            action_is_pad = batch["action_is_pad"]
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

        image_streams = self._extract_policy_image_streams(frames)

        out = self.backend.loss_from_batch(
            FoundationBatch(
                image_streams=image_streams,
                image_padding_masks=self._extract_policy_image_padding_masks(image_streams),
                task_text=instructions,
                target_codes=codes,
                target_latent_vectors=vectors,
                target_actions=actions,
                action_is_pad=action_is_pad,
                state=state,
            ),
            mode=self.backend_mode,
        )
        return out, codes, vectors, actions, frames, instructions

    @torch.no_grad()
    def _predict_freeform_text(
        self,
        *,
        frames: torch.Tensor,
        instructions: list[str],
        max_new_tokens: int = 32,
    ) -> list[str]:
        model = self.vla_model
        processor = self.processor
        chat = self.chat
        frames_to_images = self.frames_to_images
        if model is None or processor is None or chat is None or frames_to_images is None:
            raise RuntimeError("Backend must expose vla_model, processor, chat, and frames_to_images for freeform.")

        from foundation.vla_inputs import build_prompt_inputs

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
