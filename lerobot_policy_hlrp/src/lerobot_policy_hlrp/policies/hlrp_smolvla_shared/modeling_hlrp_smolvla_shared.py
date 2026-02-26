from __future__ import annotations

from collections import deque
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from foundation.action_tokens import ActionTokenConfig
from foundation.backends.interfaces import FoundationBatch
from foundation.backends.smolvla_shared.artifact import (
    SmolVLASharedArtifactManifest,
    load_smolvla_shared_artifact,
)
from foundation.backends.smolvla_shared.config import SmolVLASharedCoreConfig
from foundation.backends.smolvla_shared.model import SmolVLASharedCore
from foundation.backends.smolvla_shared.preprocess import pad_vector
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
        self.dataset_meta = dataset_meta

        self._image_key = self._infer_image_key(config)
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
                torch_dtype=_dtype_from_name(self.config.torch_dtype),
                trust_remote_code=bool(self.config.trust_remote_code),
                action_tokens=ActionTokenConfig(
                    codebook_size=int(self.config.codebook_size),
                    code_seq_len=int(self.config.code_seq_len),
                ),
                use_gpu_preprocessing=bool(self.config.use_gpu_preprocessing),
                image_size=tuple(self.config.image_size),
                flow_hidden_dim=int(self.config.flow_hidden_dim),
                flow_steps=int(self.config.flow_steps),
                min_period=float(self.config.min_period),
                max_period=float(self.config.max_period),
                time_beta_alpha=float(self.config.time_beta_alpha),
                time_beta_beta=float(self.config.time_beta_beta),
            )
        )

        self.core.setup(device=torch.device(str(self.config.device)))
        self._try_load_stage2_artifact(stage2_artifact_path=self.config.stage2_artifact)

        self._queues: dict[str, deque[torch.Tensor]] = {}
        self.reset()

    @staticmethod
    def _infer_image_key(config: HLRPSmolVLASharedConfig) -> str:
        keys = list(config.image_features.keys())
        if not keys:
            raise ValueError("Policy requires at least one image feature.")
        return keys[0]

    @staticmethod
    def _infer_action_dim(config: HLRPSmolVLASharedConfig) -> int:
        feature = config.action_feature
        if feature is None:
            raise ValueError("Missing action feature.")
        dim = 1
        for s in feature.shape:
            dim *= int(s)
        return int(dim)

    def _extract_frames(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._image_key not in batch:
            raise KeyError(f"Expected image key '{self._image_key}' in batch.")

        img = batch[self._image_key]
        if img.ndim == 4:
            if img.shape[-1] == 3:
                return img.unsqueeze(1)
            if img.shape[1] == 3:
                return img.unsqueeze(1)
            raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

        if img.ndim == 5:
            if img.shape[-1] == 3:
                return img[:, -1:, ...]
            if img.shape[2] == 3:
                return img[:, -1:, ...]
            if img.shape[1] == 3:
                return img[:, :, -1:, ...]
            raise ValueError(f"Unsupported image sequence shape: {tuple(img.shape)}")

        raise ValueError(f"Expected 4D/5D image tensor, got shape {tuple(img.shape)}")

    def _extract_instructions(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        task = batch.get("task")
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, list):
            if len(task) != batch_size:
                raise ValueError(f"task list length mismatch: expected {batch_size}, got {len(task)}")
            return [str(t) for t in task]
        return [""] * batch_size

    def _extract_action_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if ACTION not in batch:
            raise KeyError("Expected 'action' in batch.")
        action = batch[ACTION]
        if action.ndim == 3:
            action = action[:, 0, :]
        elif action.ndim != 2:
            raise ValueError(f"Expected 2D or 3D action tensor, got shape {tuple(action.shape)}")
        return pad_vector(action, int(self.config.max_action_dim))

    def _to_foundation_batch(self, batch: dict[str, Any]) -> FoundationBatch:
        frames = self._extract_frames(batch)
        instructions = self._extract_instructions(batch, batch_size=int(frames.shape[0]))
        state = batch.get(OBS_STATE)
        if state is not None:
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32)
            state = state.to(torch.float32)
        return FoundationBatch(frames=frames, instructions=instructions, state=state)

    def _try_load_stage2_artifact(
        self,
        *,
        stage2_artifact_path: Path | None,
    ) -> None:
        if stage2_artifact_path is None:
            return

        artifact_path = Path(stage2_artifact_path)
        manifest, core_state_dict = load_smolvla_shared_artifact(
            path=artifact_path,
        )
        self._assert_manifest_compatible(manifest=manifest)

        missing_allowed = {
            "action_head.weight",
            "action_head.bias",
            "action_out_proj.weight",
            "action_out_proj.bias",
        }
        missing, unexpected = self.core.load_state_dict(core_state_dict, strict=False)
        missing_disallowed = [k for k in missing if k not in missing_allowed]
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys while loading stage2 artifact {artifact_path}: {list(unexpected)}"
            )
        if missing_disallowed:
            raise RuntimeError(
                f"Missing required keys while loading stage2 artifact {artifact_path}: {missing_disallowed}"
            )
        logger.info(
            "Loaded stage2 artifact %s (schema=%s, source_run=%s, source_step=%s, missing_optional=%s)",
            artifact_path,
            manifest.schema_version,
            manifest.source_run_dir,
            manifest.source_global_step,
            sorted(list(missing)),
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
        if int(self.config.latent_vector_dim) != int(manifest.latent_vector_dim):
            mismatches.append(
                f"latent_vector_dim config={self.config.latent_vector_dim} artifact={manifest.latent_vector_dim}"
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
        foundation_batch = self._to_foundation_batch(batch)
        pred_action = self.core.predict_actions(batch=foundation_batch)
        target_action = self._extract_action_target(batch).to(device=pred_action.device, dtype=pred_action.dtype)
        loss = F.mse_loss(pred_action, target_action)
        return loss, {"loss": float(loss.detach().cpu())}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        foundation_batch = self._to_foundation_batch(batch)
        pred_action = self.core.predict_actions(batch=foundation_batch)
        pred_action = pred_action[:, : self._action_dim]
        return pred_action.to(dtype=torch.float32).unsqueeze(1).expand(-1, self.config.n_action_steps, -1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if len(self._queues[ACTION]) == 0:
            chunk = self.predict_action_chunk(batch)
            for idx in range(self.config.n_action_steps):
                self._queues[ACTION].append(chunk[:, idx, :])

        return self._queues[ACTION].popleft()
