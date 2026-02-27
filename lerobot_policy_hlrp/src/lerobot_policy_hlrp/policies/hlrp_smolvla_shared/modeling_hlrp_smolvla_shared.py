from __future__ import annotations

from collections import deque
from logging import getLogger
from pathlib import Path
from typing import Any

import torch

from foundation.action_tokens import ActionTokenConfig
from foundation.backends.interfaces import FoundationBatch
from foundation.backends.smolvla_shared.artifact import (
    SmolVLASharedArtifactManifest,
    load_smolvla_shared_artifact,
)
from foundation.backends.smolvla_shared.config import SmolVLASharedCoreConfig
from foundation.backends.smolvla_shared.input_transform import (
    resolve_action_pad_field,
    resolve_action_pad_mask,
    to_action_chunk,
)
from foundation.backends.smolvla_shared.input_transform import (
    normalize_vector_mean_std,
    unnormalize_vector_mean_std,
)
from foundation.backends.smolvla_shared.model import SmolVLASharedCore
from foundation.backends.smolvla_shared.preprocess import pad_vector
from foundation.vla_inputs import ChatConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

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
    """LeRobot policy adapter for the shared SmolVLA implementation in packages/foundation."""

    config_class = HLRPSmolVLASharedConfig
    name = "hlrp_smolvla_shared"

    def __init__(
        self,
        config: HLRPSmolVLASharedConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        dataset_meta=None,
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
        if self.config.init_mode == "artifact":
            if self.config.stage2_artifact is None:
                raise RuntimeError("init_mode='artifact' requires stage2_artifact")
            self._try_load_stage2_artifact(stage2_artifact_path=self.config.stage2_artifact)
        else:
            logger.info("Initialized policy in scratch mode (no stage2 artifact load).")

        self._queues: dict[str, deque[torch.Tensor]] = {}
        self.reset()

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

    def _extract_image_padding_masks(
        self,
        batch: dict[str, torch.Tensor],
        *,
        image_streams: dict[str, torch.Tensor],
        require_image_padding_masks: bool,
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for key in image_streams:
            is_pad_key = f"{key}_is_pad"
            if is_pad_key in batch:
                masks[key] = ~batch[is_pad_key].to(dtype=torch.bool)
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

    def _to_foundation_batch(
        self,
        batch: dict[str, Any],
        *,
        require_action_is_pad: bool,
        require_image_padding_masks: bool,
    ) -> FoundationBatch:
        image_streams = self._extract_image_streams(batch)
        first_key = next(iter(image_streams))
        batch_size = int(image_streams[first_key].shape[0])
        instructions = self._extract_instructions(batch, batch_size=batch_size)
        state = batch[OBS_STATE]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(torch.float32)
        state = normalize_vector_mean_std(
            value=state,
            stats=self.normalization_stats,
            key_candidates=[OBS_STATE, "observation.state"],
        )
        action_is_pad = None
        if require_action_is_pad:
            action_is_pad = self._extract_action_is_pad(batch, batch_size=batch_size)
        return FoundationBatch(
            image_streams=image_streams,
            image_padding_masks=self._extract_image_padding_masks(
                batch,
                image_streams=image_streams,
                require_image_padding_masks=require_image_padding_masks,
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

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        foundation_batch = self._to_foundation_batch(
            batch, require_action_is_pad=True, require_image_padding_masks=True
        )
        target_action = self._extract_action_target(batch)
        loss = self.core.action_flow_loss(
            batch=foundation_batch,
            target_actions=target_action,
            action_is_pad=foundation_batch.action_is_pad,
        )
        return loss, {"loss": float(loss.detach().cpu())}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        foundation_batch = self._to_foundation_batch(
            batch, require_action_is_pad=False, require_image_padding_masks=False
        )
        pred_chunk = self.core.sample_action_chunk(batch=foundation_batch)
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
