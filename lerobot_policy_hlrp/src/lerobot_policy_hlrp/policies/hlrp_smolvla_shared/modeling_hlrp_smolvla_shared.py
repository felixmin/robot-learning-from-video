from __future__ import annotations

from collections import deque
import json
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from stage2.action_tokens import ActionTokenConfig
from stage2.backends.interfaces import Stage2Batch
from stage2.backends.smolvla_shared.artifact import (
    SmolVLASharedArtifactManifest,
    load_smolvla_shared_artifact,
)
from stage2.backends.smolvla_shared.config import SmolVLASharedCoreConfig
from stage2.backends.smolvla_shared.input_transform import (
    resolve_action_pad_field,
    resolve_action_pad_mask,
    to_action_chunk,
)
from stage2.backends.smolvla_shared.input_transform import (
    normalize_vector_mean_std,
    unnormalize_vector_mean_std,
)
from stage2.backends.smolvla_shared.model import SmolVLASharedCore
from stage2.backends.smolvla_shared.preprocess import pad_vector
from stage2.policy_inputs import ChatConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.checkpoint_stats import (
    NORMALIZATION_STATS_FILENAME,
    require_saved_normalization_stats,
    write_normalization_stats,
)
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.configuration_hlrp_smolvla_shared import (
    HLRPSmolVLASharedConfig,
)


logger = getLogger(__name__)


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if key == "fp32":
        return torch.float32
    raise ValueError(f"Unknown torch dtype name: {name!r}")


class HLRPSmolVLASharedPolicy(PreTrainedPolicy):
    """LeRobot policy adapter for the shared SmolVLA implementation in packages/stage2."""

    config_class = HLRPSmolVLASharedConfig
    name = "hlrp_smolvla_shared"
    _LATENT_TRAINING_MODES = {"latent", "multitask", "alternating"}
    _ACTION_SUPERVISION_KEY = "hlrp_action_supervised"
    _LATENT_SUPERVISION_KEY = "hlrp_latent_supervised"
    _SOURCE_NAME_KEY = "hlrp_source_name"

    def __init__(
        self,
        config: HLRPSmolVLASharedConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        dataset_meta=None,
        load_stage2_artifact: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()

        self.config = config
        self.dataset_stats = dataset_stats
        self.normalization_stats = dataset_stats
        self.dataset_meta = dataset_meta

        self._image_keys = self._resolve_image_keys(config)
        self._action_dim = self._infer_action_dim(config)
        if int(self.config.max_action_dim) < int(self._action_dim):
            raise ValueError(
                f"max_action_dim ({self.config.max_action_dim}) must be >= action_dim ({self._action_dim})"
            )

        self.core = SmolVLASharedCore(
            config=SmolVLASharedCoreConfig(
                model_name=str(self.config.model_name),
                latent_vector_dim=int(self.config.latent_vector_dim),
                action_dim=int(self.config.max_action_dim),
                action_chunk_size=int(self.config.chunk_size),
                freeze_vlm=bool(self.config.train_expert_only),
                freeze_vision_encoder=bool(self.config.freeze_vision_encoder),
                load_vlm_weights=bool(self.config.load_vlm_weights),
                attention_mode=str(self.config.attention_mode),
                num_expert_layers=int(self.config.num_expert_layers),
                num_vlm_layers=int(self.config.num_vlm_layers),
                self_attn_every_n_layers=int(self.config.self_attn_every_n_layers),
                expert_width_multiplier=float(self.config.expert_width_multiplier),
                add_image_special_tokens=bool(self.config.add_image_special_tokens),
                prefix_length=int(self.config.prefix_length),
                torch_dtype=_dtype_from_name(self.config.torch_dtype),
                trust_remote_code=bool(self.config.trust_remote_code),
                chat=ChatConfig(system_prompt=self.config.system_prompt),
                action_tokens=ActionTokenConfig(
                    codebook_size=int(self.config.codebook_size),
                    code_seq_len=int(self.config.code_seq_len),
                ),
                use_gpu_preprocessing=bool(self.config.use_gpu_preprocessing),
                image_size=tuple(self.config.image_size),
                camera_keys=tuple(self._image_keys),
                empty_cameras=int(self.config.empty_cameras),
                tokenizer_max_length=int(self.config.tokenizer_max_length),
                pad_language_to=str(self.config.pad_language_to),
                max_state_dim=int(self.config.max_state_dim),
                flow_hidden_dim=int(self.config.flow_hidden_dim),
                flow_steps=int(self.config.flow_steps),
                min_period=float(self.config.min_period),
                max_period=float(self.config.max_period),
                time_beta_alpha=float(self.config.time_beta_alpha),
                time_beta_beta=float(self.config.time_beta_beta),
            )
        )

        self.core.setup(device=torch.device(str(self.config.device)))
        if self.config.init_mode == "artifact" and load_stage2_artifact:
            if self.config.stage2_artifact is None:
                raise RuntimeError("init_mode='artifact' requires stage2_artifact")
            self._try_load_stage2_artifact(stage2_artifact_path=self.config.stage2_artifact)
        elif self.config.init_mode == "artifact":
            logger.info(
                "Skipping stage2 artifact initialization while loading checkpoint weights from %s.",
                self.config.pretrained_path,
            )
        else:
            logger.info("Initialized policy in scratch mode (no stage2 artifact load).")

        self._stage1_teacher = None
        self._stage1_image_size: tuple[int, int] | None = None
        self._train_update_calls = 0
        self._active_training_mode: str | None = None
        self._queues: dict[str, deque[torch.Tensor]] = {}
        self.reset()

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        if self.normalization_stats is None:
            raise RuntimeError(
                "Refusing to save HLRP checkpoint without normalization stats. "
                f"Expected {NORMALIZATION_STATS_FILENAME} to be written alongside the checkpoint."
            )
        write_normalization_stats(
            save_directory=save_directory,
            stats=self.normalization_stats,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs):
        kwargs.setdefault("load_stage2_artifact", False)
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        saved_stats = require_saved_normalization_stats(
            pretrained_name_or_path,
            init_mode=str(policy.config.init_mode),
            force_download=kwargs.get("force_download", False),
            resume_download=kwargs.get("resume_download"),
            proxies=kwargs.get("proxies"),
            token=kwargs.get("token"),
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only", False),
            revision=kwargs.get("revision"),
        )
        if saved_stats is not None:
            policy.dataset_stats = saved_stats
            policy.normalization_stats = saved_stats
        return policy

    @staticmethod
    def _resolve_image_keys(config: HLRPSmolVLASharedConfig) -> list[str]:
        if config.camera_keys is not None:
            keys = [str(k) for k in config.camera_keys]
        else:
            keys = [str(k) for k in config.image_features.keys()]
        if not keys:
            raise ValueError("Policy requires at least one image feature.")
        return keys

    @staticmethod
    def _infer_action_dim(config: HLRPSmolVLASharedConfig) -> int:
        feature = config.action_feature
        if feature is None:
            raise ValueError("Missing action feature.")
        dim = 1
        for s in feature.shape:
            dim *= int(s)
        return int(dim)

    def _uses_latent_targets(self) -> bool:
        return str(self.config.stage3_training_mode) in self._LATENT_TRAINING_MODES

    def _peek_training_mode_for_step(self) -> str:
        mode = str(self.config.stage3_training_mode)
        if mode != "alternating":
            return mode
        latent_steps = int(self.config.alternating_latent_steps_per_action_step)
        cycle = latent_steps + 1
        step = int(getattr(self, "_train_update_calls", 0))
        if (step % cycle) < latent_steps:
            return "latent"
        return "multitask"

    def _training_mode_for_step(self) -> str:
        active_training_mode = getattr(self, "_active_training_mode", None)
        if active_training_mode is not None:
            return active_training_mode
        return self._peek_training_mode_for_step()

    def begin_training_step(self) -> str:
        mode = self._peek_training_mode_for_step()
        self._active_training_mode = mode
        return mode

    def end_training_step(self) -> None:
        active_training_mode = getattr(self, "_active_training_mode", None)
        if active_training_mode is None:
            return
        if str(self.config.stage3_training_mode) == "alternating":
            self._train_update_calls = int(getattr(self, "_train_update_calls", 0)) + 1
        self._active_training_mode = None

    def _supervision_mask(
        self,
        batch: dict[str, Any],
        *,
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"Stage-3 supervision key {key!r} is missing from batch")
        values = batch[key]
        if torch.is_tensor(values):
            mask = values.to(device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(values, device=device, dtype=torch.bool)
        mask = mask.reshape(batch_size)
        return mask

    @staticmethod
    def _sanitize_metric_suffix(value: str) -> str:
        out = []
        for char in str(value):
            out.append(char if char.isalnum() else "_")
        return "".join(out).strip("_") or "unknown"

    def _source_mix_metrics(self, batch: dict[str, Any], *, batch_size: int) -> dict[str, float]:
        raw = batch.get(self._SOURCE_NAME_KEY)
        if raw is None:
            return {}
        if isinstance(raw, str):
            names = [raw] * batch_size
        elif isinstance(raw, list):
            if len(raw) != batch_size:
                raise ValueError(
                    f"{self._SOURCE_NAME_KEY} length mismatch: expected {batch_size}, got {len(raw)}"
                )
            names = [str(value) for value in raw]
        else:
            return {}

        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        total = float(batch_size)
        return {
            f"source_frac_{self._sanitize_metric_suffix(name)}": float(count) / total
            for name, count in counts.items()
        }

    def _conditioning_step_index(self) -> int:
        return 0 if self._uses_latent_targets() else -1

    def _extract_image_streams(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        streams: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            if key in batch:
                streams[key] = batch[key]
        if not streams:
            raise KeyError(
                f"None of the configured image keys were found. configured={self._image_keys}, batch={list(batch.keys())}"
            )
        return streams

    @staticmethod
    def _select_observation_step(tensor: torch.Tensor, *, step_index: int) -> torch.Tensor:
        if tensor.ndim != 5:
            return tensor
        idx = int(step_index)
        if idx < 0:
            idx = int(tensor.shape[1]) + idx
        if idx < 0 or idx >= int(tensor.shape[1]):
            raise IndexError(
                f"Observation step index {step_index} out of bounds for tensor with T={int(tensor.shape[1])}"
            )
        return tensor[:, idx, ...]

    def _extract_conditioning_streams(
        self, image_streams: dict[str, torch.Tensor], *, step_index: int
    ) -> dict[str, torch.Tensor]:
        return {
            key: self._select_observation_step(stream, step_index=step_index)
            for key, stream in image_streams.items()
        }

    def _extract_image_padding_masks(
        self,
        batch: dict[str, torch.Tensor],
        *,
        image_streams: dict[str, torch.Tensor],
        require_image_padding_masks: bool,
        conditioning_step_index: int,
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for key in image_streams:
            is_pad_key = f"{key}_is_pad"
            if is_pad_key in batch:
                mask = ~batch[is_pad_key].to(dtype=torch.bool)
                if mask.ndim == 2:
                    idx = int(conditioning_step_index)
                    if idx < 0:
                        idx = int(mask.shape[1]) + idx
                    if idx < 0 or idx >= int(mask.shape[1]):
                        raise IndexError(
                            f"Image padding mask index {conditioning_step_index} out of bounds for key={key!r} with T={int(mask.shape[1])}"
                        )
                    mask = mask[:, idx]
                masks[key] = mask
                continue
            if require_image_padding_masks:
                raise KeyError(is_pad_key)
            stream = image_streams[key]
            masks[key] = torch.ones((int(stream.shape[0]),), dtype=torch.bool, device=stream.device)
        return masks

    def _extract_instructions(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        task = batch["task"]
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, list):
            if len(task) != batch_size:
                raise ValueError(f"task list length mismatch: expected {batch_size}, got {len(task)}")
            return [str(t) for t in task]
        raise TypeError(f"Expected task as str or list[str], got {type(task)}")

    def _extract_action_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if ACTION not in batch:
            raise KeyError("Expected 'action' in batch.")
        action = batch[ACTION]
        action = to_action_chunk(actions=action, chunk_size=int(self.config.chunk_size))
        action = pad_vector(action, int(self.config.max_action_dim))
        return normalize_vector_mean_std(
            value=action,
            stats=self.normalization_stats,
            key_candidates=[ACTION, "action"],
        )

    def _extract_action_is_pad(self, batch: dict[str, Any], *, batch_size: int) -> torch.Tensor:
        mask = resolve_action_pad_field(
            batch=batch,
            action_is_pad_key="action_is_pad",
            actions_id_pad_key="actions_id_pad",
        )
        device = next(self.core.parameters()).device
        return resolve_action_pad_mask(
            action_is_pad=mask,
            batch_size=batch_size,
            chunk_size=int(self.config.chunk_size),
            device=device,
        )

    def _extract_stage1_frame_pairs(
        self,
        batch: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.config.lam_camera_keys is None:
            raise RuntimeError("lam_camera_keys must be set for latent supervision modes")
        frame_pairs: dict[str, torch.Tensor] = {}
        valid_pair: torch.Tensor | None = None
        for key in self.config.lam_camera_keys:
            if key not in batch:
                raise KeyError(f"Stage-1 LAM camera key {key!r} is missing from batch")
            frames = batch[key]
            if not torch.is_tensor(frames):
                raise TypeError(f"Expected tensor for camera key {key!r}, got {type(frames)}")
            if frames.ndim != 5:
                raise ValueError(
                    f"Expected camera tensor [B,2,C,H,W] or [B,2,H,W,C] for key {key!r}, got {tuple(frames.shape)}"
                )
            if int(frames.shape[1]) != 2:
                raise ValueError(
                    f"Expected exactly 2 observation steps for Stage-1 LAM key {key!r}, got T={int(frames.shape[1])}"
                )
            if int(frames.shape[2]) == 3:
                frames_t = frames
            elif int(frames.shape[-1]) == 3:
                frames_t = frames.permute(0, 1, 4, 2, 3)
            else:
                raise ValueError(f"Unsupported Stage-1 LAM frame layout for key {key!r}: {tuple(frames.shape)}")
            if frames_t.dtype == torch.uint8:
                frames_t = frames_t.to(torch.float32) / 255.0
            else:
                frames_t = frames_t.to(torch.float32)
            frame_pairs[str(key)] = frames_t

            is_pad_key = f"{key}_is_pad"
            if is_pad_key not in batch:
                raise KeyError(is_pad_key)
            is_pad = batch[is_pad_key]
            if not torch.is_tensor(is_pad):
                is_pad = torch.as_tensor(is_pad, dtype=torch.bool, device=frames_t.device)
            is_pad = is_pad.to(device=frames_t.device, dtype=torch.bool)
            if is_pad.ndim != 2 or int(is_pad.shape[1]) != 2:
                raise ValueError(
                    f"Expected {is_pad_key!r} with shape [B,2], got {tuple(is_pad.shape)}"
                )
            keep = (~is_pad[:, 0]) & (~is_pad[:, 1])
            valid_pair = keep if valid_pair is None else (valid_pair & keep)

        if valid_pair is None:
            raise RuntimeError("Stage-1 LAM camera extraction produced no camera streams.")
        return frame_pairs, valid_pair

    def _extract_stage1_valid_pair(self, batch: dict[str, Any]) -> torch.Tensor:
        _, valid_pair = self._extract_stage1_frame_pairs(batch)
        return valid_pair

    def get_accumulation_denominators(self, batch: dict[str, Any]) -> dict[str, float]:
        active_mode = self._training_mode_for_step()
        denominators = {"action": 0.0, "latent": 0.0}

        if active_mode in {"action", "multitask"}:
            target_action = self._extract_action_target(batch)
            batch_size = int(target_action.shape[0])
            action_keep = self._supervision_mask(
                batch,
                key=self._ACTION_SUPERVISION_KEY,
                batch_size=batch_size,
                device=target_action.device,
            )
            if bool(action_keep.any().item()):
                action_is_pad = self._extract_action_is_pad(batch, batch_size=batch_size)[action_keep]
                valid_steps = (~action_is_pad).sum()
                denominators["action"] = float(valid_steps.item() * int(target_action.shape[-1]))

        if active_mode in {"latent", "multitask"}:
            valid_pair = self._extract_stage1_valid_pair(batch).to(dtype=torch.bool)
            keep = valid_pair & self._supervision_mask(
                batch,
                key=self._LATENT_SUPERVISION_KEY,
                batch_size=int(valid_pair.shape[0]),
                device=valid_pair.device,
            )
            denominators["latent"] = float(keep.sum().item() * int(self.config.latent_vector_dim))

        return denominators

    def _ensure_stage1_teacher(self) -> None:
        if self._stage1_teacher is not None:
            return
        if self.config.lam_checkpoint_path is None:
            raise RuntimeError("lam_checkpoint_path must be set for latent supervision modes")
        from lam import load_lam_encoder_vq_inference_from_checkpoint

        device = next(self.core.parameters()).device
        teacher = load_lam_encoder_vq_inference_from_checkpoint(
            checkpoint_path=str(self.config.lam_checkpoint_path),
            map_location=device,
            strict=True,
            prune_decoders=True,
        )
        teacher.to(device=device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        latent_dim = int(teacher.code_seq_len) * int(teacher.codebook_dim)
        if latent_dim != int(self.config.latent_vector_dim):
            raise ValueError(
                f"Stage-1 LAM latent dim mismatch: config.latent_vector_dim={self.config.latent_vector_dim} "
                f"but teacher provides {latent_dim} (code_seq_len={teacher.code_seq_len}, codebook_dim={teacher.codebook_dim})"
            )
        if int(teacher.code_seq_len) != int(self.config.code_seq_len):
            raise ValueError(
                f"Stage-1 LAM code_seq_len mismatch: config.code_seq_len={self.config.code_seq_len} "
                f"teacher.code_seq_len={teacher.code_seq_len}"
            )

        self._stage1_teacher = teacher
        self._stage1_image_size = tuple(int(x) for x in teacher.image_size)

    def _compute_latent_targets_online(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_stage1_teacher()
        if self._stage1_teacher is None or self._stage1_image_size is None:
            raise RuntimeError("Stage-1 LAM teacher is not initialized.")

        frame_pairs, valid_pair = self._extract_stage1_frame_pairs(batch)
        if len(frame_pairs) != 1:
            raise NotImplementedError(
                f"Current Stage-3 Stage-1 LAM path supports exactly one camera key, got {tuple(frame_pairs.keys())}"
            )
        pair = next(iter(frame_pairs.values()))
        b, t, c, h, w = pair.shape
        if (int(h), int(w)) != self._stage1_image_size:
            pair = F.interpolate(
                pair.reshape(b * t, c, h, w),
                size=self._stage1_image_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, self._stage1_image_size[0], self._stage1_image_size[1])
        video = pair.permute(0, 2, 1, 3, 4)
        _, vectors = self._stage1_teacher.codes_and_vectors_from_video(video)
        vectors = vectors.reshape(vectors.shape[0], -1)
        if int(vectors.shape[1]) != int(self.config.latent_vector_dim):
            raise ValueError(
                f"Latent target dim mismatch: expected {self.config.latent_vector_dim}, got {int(vectors.shape[1])}"
            )
        return vectors, valid_pair

    @staticmethod
    def _slice_stage2_batch(batch: Stage2Batch, keep: torch.Tensor) -> Stage2Batch:
        if keep.ndim != 1:
            raise ValueError(f"keep mask must be rank 1, got {tuple(keep.shape)}")
        keep = keep.to(dtype=torch.bool)

        image_streams = (
            {k: v[keep] for k, v in batch.image_streams.items()} if batch.image_streams is not None else None
        )
        image_padding_masks = (
            {k: v[keep] for k, v in batch.image_padding_masks.items()}
            if batch.image_padding_masks is not None
            else None
        )

        task_text = None
        keep_cpu = keep.detach().cpu().tolist()
        if batch.task_text is not None:
            task_text = [text for text, flag in zip(batch.task_text, keep_cpu) if flag]

        return Stage2Batch(
            image_streams=image_streams,
            image_padding_masks=image_padding_masks,
            task_text=task_text,
            subtask_text=(
                None if batch.subtask_text is None else [x for x, f in zip(batch.subtask_text, keep_cpu) if f]
            ),
            language_tokens=None if batch.language_tokens is None else batch.language_tokens[keep],
            language_attention_mask=None
            if batch.language_attention_mask is None
            else batch.language_attention_mask[keep],
            target_codes=None if batch.target_codes is None else batch.target_codes[keep],
            target_latent_vectors=(
                None if batch.target_latent_vectors is None else batch.target_latent_vectors[keep]
            ),
            target_actions=None if batch.target_actions is None else batch.target_actions[keep],
            action_is_pad=None if batch.action_is_pad is None else batch.action_is_pad[keep],
            state=None if batch.state is None else batch.state[keep],
            meta=batch.meta,
        )

    def _to_stage2_batch(
        self,
        batch: dict[str, Any],
        *,
        require_action_is_pad: bool,
        require_image_padding_masks: bool,
        conditioning_step_index: int,
    ) -> Stage2Batch:
        raw_streams = self._extract_image_streams(batch)
        image_streams = self._extract_conditioning_streams(
            raw_streams,
            step_index=conditioning_step_index,
        )
        first_key = next(iter(image_streams))
        batch_size = int(image_streams[first_key].shape[0])
        instructions = self._extract_instructions(batch, batch_size=batch_size)
        state = batch[OBS_STATE]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 3:
            idx = int(conditioning_step_index)
            if idx < 0:
                idx = int(state.shape[1]) + idx
            if idx < 0 or idx >= int(state.shape[1]):
                raise IndexError(
                    f"State step index {conditioning_step_index} out of bounds for state tensor with T={int(state.shape[1])}"
                )
            state = state[:, idx, :]
        state = state.to(torch.float32)
        state = normalize_vector_mean_std(
            value=state,
            stats=self.normalization_stats,
            key_candidates=[OBS_STATE, "observation.state"],
        )
        action_is_pad = None
        if require_action_is_pad:
            action_is_pad = self._extract_action_is_pad(batch, batch_size=batch_size)
        return Stage2Batch(
            image_streams=image_streams,
            image_padding_masks=self._extract_image_padding_masks(
                batch,
                image_streams=image_streams,
                require_image_padding_masks=require_image_padding_masks,
                conditioning_step_index=conditioning_step_index,
            ),
            task_text=instructions,
            state=state,
            action_is_pad=action_is_pad,
        )

    def _try_load_stage2_artifact(
        self,
        *,
        stage2_artifact_path: Path,
    ) -> None:
        artifact_path = Path(stage2_artifact_path)
        manifest, core_state_dict = load_smolvla_shared_artifact(
            path=artifact_path,
        )
        self._assert_manifest_compatible(manifest=manifest)
        self.normalization_stats = manifest.normalization_stats

        missing, unexpected = self.core.load_state_dict(core_state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Strict load failed for stage2 artifact {artifact_path}. missing={list(missing)} unexpected={list(unexpected)}"
            )
        logger.info(
            "Loaded stage2 artifact %s (schema=%s, source_run=%s, source_step=%s)",
            artifact_path,
            manifest.schema_version,
            manifest.source_run_dir,
            manifest.source_global_step,
        )

    def _assert_manifest_compatible(self, *, manifest: SmolVLASharedArtifactManifest) -> None:
        mismatches: list[str] = []
        if str(self.config.model_name) != str(manifest.model_name):
            mismatches.append(
                f"model_name config={self.config.model_name!r} artifact={manifest.model_name!r}"
            )
        if str(self.config.torch_dtype).lower() != str(manifest.torch_dtype).lower():
            mismatches.append(
                f"torch_dtype config={self.config.torch_dtype!r} artifact={manifest.torch_dtype!r}"
            )
        if tuple(self.config.image_size) != tuple(manifest.image_size):
            mismatches.append(
                f"image_size config={tuple(self.config.image_size)!r} artifact={tuple(manifest.image_size)!r}"
            )
        if int(self.config.max_action_dim) != int(manifest.action_dim):
            mismatches.append(
                f"action_dim config={self.config.max_action_dim} artifact={manifest.action_dim}"
            )
        if int(self.config.chunk_size) != int(manifest.action_chunk_size):
            mismatches.append(
                f"action_chunk_size config={self.config.chunk_size} artifact={manifest.action_chunk_size}"
            )
        if int(self.config.code_seq_len) != int(manifest.code_seq_len):
            mismatches.append(
                f"code_seq_len config={self.config.code_seq_len} artifact={manifest.code_seq_len}"
            )
        if int(self.config.latent_vector_dim) != int(manifest.latent_vector_dim):
            mismatches.append(
                f"latent_vector_dim config={self.config.latent_vector_dim} artifact={manifest.latent_vector_dim}"
            )
        if int(self.config.tokenizer_max_length) != int(manifest.tokenizer_max_length):
            mismatches.append(
                "tokenizer_max_length "
                f"config={self.config.tokenizer_max_length} artifact={manifest.tokenizer_max_length}"
            )
        if str(self.config.pad_language_to) != str(manifest.pad_language_to):
            mismatches.append(
                f"pad_language_to config={self.config.pad_language_to!r} artifact={manifest.pad_language_to!r}"
            )
        if str(self.config.system_prompt or "") != str(manifest.system_prompt or ""):
            mismatches.append(
                f"system_prompt config={self.config.system_prompt!r} artifact={manifest.system_prompt!r}"
            )
        if int(self.config.max_state_dim) != int(manifest.max_state_dim):
            mismatches.append(
                f"max_state_dim config={self.config.max_state_dim} artifact={manifest.max_state_dim}"
            )
        cfg_camera_keys = tuple(self.config.camera_keys) if self.config.camera_keys is not None else None
        manifest_camera_keys = (
            tuple(manifest.camera_keys) if manifest.camera_keys is not None else None
        )
        if cfg_camera_keys != manifest_camera_keys:
            mismatches.append(
                f"camera_keys config={cfg_camera_keys!r} artifact={manifest_camera_keys!r}"
            )
        if int(self.config.flow_hidden_dim) != int(manifest.flow_hidden_dim):
            mismatches.append(
                f"flow_hidden_dim config={self.config.flow_hidden_dim} artifact={manifest.flow_hidden_dim}"
            )
        if int(self.config.flow_steps) != int(manifest.flow_steps):
            mismatches.append(
                f"flow_steps config={self.config.flow_steps} artifact={manifest.flow_steps}"
            )
        if float(self.config.min_period) != float(manifest.min_period):
            mismatches.append(
                f"min_period config={self.config.min_period} artifact={manifest.min_period}"
            )
        if float(self.config.max_period) != float(manifest.max_period):
            mismatches.append(
                f"max_period config={self.config.max_period} artifact={manifest.max_period}"
            )
        if float(self.config.time_beta_alpha) != float(manifest.time_beta_alpha):
            mismatches.append(
                f"time_beta_alpha config={self.config.time_beta_alpha} artifact={manifest.time_beta_alpha}"
            )
        if float(self.config.time_beta_beta) != float(manifest.time_beta_beta):
            mismatches.append(
                f"time_beta_beta config={self.config.time_beta_beta} artifact={manifest.time_beta_beta}"
            )
        if mismatches:
            raise ValueError(
                "Stage2 artifact manifest is incompatible with policy config: "
                + "; ".join(mismatches)
            )

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _zero_loss(self) -> torch.Tensor:
        for param in self.core.parameters():
            if param.requires_grad:
                return param.sum() * 0.0
        param = next(self.core.parameters(), None)
        if param is not None:
            return torch.zeros((), device=param.device, dtype=param.dtype, requires_grad=True)
        return torch.zeros((), requires_grad=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        active_mode = self._training_mode_for_step()
        needs_action_targets = active_mode in {"action", "multitask"}
        stage2_batch = self._to_stage2_batch(
            batch,
            require_action_is_pad=needs_action_targets,
            require_image_padding_masks=True,
            conditioning_step_index=self._conditioning_step_index(),
        )
        batch_size = int(next(iter(stage2_batch.image_streams.values())).shape[0])
        action_loss: torch.Tensor | None = None
        latent_loss: torch.Tensor | None = None

        action_supervised_count = 0
        if active_mode in {"action", "multitask"}:
            target_action = self._extract_action_target(batch)
            action_keep = self._supervision_mask(
                batch,
                key=self._ACTION_SUPERVISION_KEY,
                batch_size=int(target_action.shape[0]),
                device=target_action.device,
            )
            action_supervised_count = int(action_keep.sum().item())
            if action_supervised_count > 0:
                action_batch = self._slice_stage2_batch(stage2_batch, action_keep)
                action_loss = self.core.action_flow_loss(
                    batch=action_batch,
                    target_actions=target_action[action_keep],
                    action_is_pad=action_batch.action_is_pad,
                )
            else:
                action_loss = self._zero_loss()

        latent_valid_pairs = 0
        latent_supervised_count = 0
        if active_mode in {"latent", "multitask"}:
            target_latent, valid_pair = self._compute_latent_targets_online(batch)
            keep = valid_pair.to(device=target_latent.device, dtype=torch.bool)
            latent_valid_pairs = int(keep.sum().item())
            keep = keep & self._supervision_mask(
                batch,
                key=self._LATENT_SUPERVISION_KEY,
                batch_size=int(keep.shape[0]),
                device=target_latent.device,
            )
            latent_supervised_count = int(keep.sum().item())
            if latent_supervised_count > 0:
                latent_batch = self._slice_stage2_batch(stage2_batch, keep)
                latent_loss = self.core.latent_flow_loss(
                    batch=latent_batch,
                    target_vectors=target_latent[keep],
                )
            else:
                latent_loss = self._zero_loss()

        if active_mode == "action":
            if action_loss is None:
                raise RuntimeError("Internal error: action_loss is None for action mode")
            total = float(self.config.action_loss_weight) * action_loss
        elif active_mode == "latent":
            if latent_loss is None:
                raise RuntimeError("Internal error: latent_loss is None for latent mode")
            total = float(self.config.latent_loss_weight) * latent_loss
        elif active_mode == "multitask":
            if action_loss is None or latent_loss is None:
                raise RuntimeError("Internal error: missing loss term for multitask mode")
            total = float(self.config.action_loss_weight) * action_loss + float(self.config.latent_loss_weight) * latent_loss
        else:
            raise ValueError(f"Unsupported active training mode: {active_mode!r}")

        metrics: dict[str, float] = {
            "loss": float(total.detach().cpu()),
            "mode_action": float(active_mode == "action"),
            "mode_latent": float(active_mode == "latent"),
            "mode_multitask": float(active_mode == "multitask"),
        }
        metrics.update(self._source_mix_metrics(batch, batch_size=batch_size))
        if action_loss is not None:
            action_loss_denom = self.get_accumulation_denominators(batch)["action"]
            metrics["action_loss"] = float(action_loss.detach().cpu())
            metrics["action_supervised_samples"] = float(action_supervised_count)
            metrics["action_supervised_fraction"] = float(action_supervised_count / int(target_action.shape[0]))
            metrics["batch_action_supervised_samples"] = float(action_supervised_count)
            metrics["batch_action_supervised_denominator"] = float(int(target_action.shape[0]))
            metrics["batch_action_supervised_fraction"] = float(
                action_supervised_count / int(target_action.shape[0])
            )
            metrics["_action_loss_tensor"] = action_loss
            metrics["_action_loss_denominator_exact"] = float(action_loss_denom)
            metrics["_action_supervised_denominator"] = float(int(target_action.shape[0]))
        if latent_loss is not None:
            latent_loss_denom = self.get_accumulation_denominators(batch)["latent"]
            metrics["latent_loss"] = float(latent_loss.detach().cpu())
            metrics["latent_valid_pairs"] = float(latent_valid_pairs)
            metrics["latent_supervised_samples"] = float(latent_supervised_count)
            metrics["latent_supervised_fraction"] = float(latent_supervised_count / int(target_latent.shape[0]))
            metrics["batch_latent_supervised_samples"] = float(latent_supervised_count)
            metrics["batch_latent_supervised_denominator"] = float(int(target_latent.shape[0]))
            metrics["batch_latent_supervised_fraction"] = float(
                latent_supervised_count / int(target_latent.shape[0])
            )
            metrics["_latent_loss_tensor"] = latent_loss
            metrics["_latent_loss_denominator_exact"] = float(latent_loss_denom)
            metrics["_latent_supervised_denominator"] = float(int(target_latent.shape[0]))
        return total, metrics

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        stage2_batch = self._to_stage2_batch(
            batch,
            require_action_is_pad=False,
            require_image_padding_masks=False,
            conditioning_step_index=self._conditioning_step_index(),
        )
        pred_chunk = self.core.sample_action_chunk(batch=stage2_batch)
        pred_chunk = pred_chunk[:, :, : self._action_dim]
        pred_chunk = pred_chunk[:, : self.config.n_action_steps, :]
        pred_chunk = unnormalize_vector_mean_std(
            value=pred_chunk,
            stats=self.normalization_stats,
            key_candidates=[ACTION, "action"],
        )
        return pred_chunk.to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if len(self._queues[ACTION]) == 0:
            chunk = self.predict_action_chunk(batch)
            for idx in range(self.config.n_action_steps):
                self._queues[ACTION].append(chunk[:, idx, :])

        return self._queues[ACTION].popleft()
