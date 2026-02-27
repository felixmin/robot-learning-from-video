from __future__ import annotations

from typing import Any

import torch

from foundation.backends.interfaces import BackendMode, FoundationBatch, LatentOutput, LossOutput
from foundation.backends.smolvla_shared.config import SmolVLASharedBackendConfig
from foundation.backends.smolvla_shared.input_transform import to_action_chunk
from foundation.backends.smolvla_shared.losses import action_flow_loss, latent_flow_loss
from foundation.backends.smolvla_shared.model import SmolVLASharedCore
from foundation.backends.smolvla_shared.smolvlm_with_expert import SmolVLMWithExpertModel


class SmolVLASharedBackend(torch.nn.Module):
    """Stage-2 backend using a shared SmolVLA-style trunk and latent flow head."""

    def __init__(
        self,
        *,
        config: SmolVLASharedBackendConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        smol_model: SmolVLMWithExpertModel | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.core = SmolVLASharedCore(
            config=config.to_core_config(),
            vlm=vlm,
            processor=processor,
            smol_model=smol_model,
        )

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)

    def setup(self, *, device: torch.device) -> None:
        self.core.setup(device=device)

    def _require_target_vector(self, batch: FoundationBatch) -> torch.Tensor:
        if batch.target_latent_vectors is None:
            raise ValueError("batch.target_latent_vectors is required")
        vec = batch.target_latent_vectors
        if vec.ndim == 3:
            vec = vec.reshape(vec.shape[0], -1)
        elif vec.ndim != 2:
            raise ValueError(f"Expected target_latent_vectors [B,S,D] or [B,D], got {tuple(vec.shape)}")
        if int(vec.shape[1]) != int(self.cfg.latent_vector_dim):
            raise ValueError(
                f"target_latent_vectors dim mismatch: expected {self.cfg.latent_vector_dim}, got {int(vec.shape[1])}"
            )
        return vec

    def _require_target_actions(self, batch: FoundationBatch) -> torch.Tensor:
        if batch.target_actions is None:
            raise ValueError("batch.target_actions is required")
        actions = batch.target_actions
        if actions.ndim not in (2, 3):
            raise ValueError(f"Expected target_actions [B,A] or [B,T,A], got {tuple(actions.shape)}")
        if self.cfg.action_dim is None:
            raise ValueError("Action mode requires action_dim in backend config")
        actions = to_action_chunk(actions=actions, chunk_size=int(self.cfg.action_chunk_size))
        if int(actions.shape[-1]) > int(self.cfg.action_dim):
            raise ValueError(
                f"target_actions dim mismatch: configured action_dim={int(self.cfg.action_dim)}, got {int(actions.shape[-1])}"
            )
        return actions

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        if mode is BackendMode.LATENT_FLOW:
            target_vec = self._require_target_vector(batch)
            latent_loss = latent_flow_loss(core=self.core, batch=batch, target_vectors=target_vec)
            total = float(self.cfg.latent_loss_weight) * latent_loss
            return LossOutput(
                loss=total,
                metrics={
                    "loss": float(total.detach().cpu().item()),
                    "latent_loss": float(latent_loss.detach().cpu().item()),
                },
            )

        if mode is BackendMode.ACTIONS:
            target_actions = self._require_target_actions(batch)
            action_loss = action_flow_loss(
                core=self.core,
                batch=batch,
                target_actions=target_actions,
                action_is_pad=batch.action_is_pad,
            )
            total = float(self.cfg.action_loss_weight) * action_loss
            return LossOutput(
                loss=total,
                metrics={
                    "loss": float(total.detach().cpu().item()),
                    "action_loss": float(action_loss.detach().cpu().item()),
                },
            )

        if mode is BackendMode.MULTITASK:
            target_vec = self._require_target_vector(batch)
            target_actions = self._require_target_actions(batch)
            latent_loss = latent_flow_loss(core=self.core, batch=batch, target_vectors=target_vec)
            action_loss = action_flow_loss(
                core=self.core,
                batch=batch,
                target_actions=target_actions,
                action_is_pad=batch.action_is_pad,
            )
            total = float(self.cfg.latent_loss_weight) * latent_loss + float(self.cfg.action_loss_weight) * action_loss
            return LossOutput(
                loss=total,
                metrics={
                    "loss": float(total.detach().cpu().item()),
                    "latent_loss": float(latent_loss.detach().cpu().item()),
                    "action_loss": float(action_loss.detach().cpu().item()),
                },
            )

        raise NotImplementedError(f"{type(self).__name__} does not support mode={mode.value!r}")

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        if mode is BackendMode.LATENT_FLOW:
            vec = self.core.sample_latent_vectors(batch=batch)
            return LatentOutput(logits=None, tokens=None, vector=vec, actions=None, meta=None)

        if mode is BackendMode.ACTIONS:
            actions = self.core.predict_actions(batch=batch)
            return LatentOutput(logits=None, tokens=None, vector=None, actions=actions, meta=None)

        if mode is BackendMode.MULTITASK:
            vec = self.core.sample_latent_vectors(batch=batch)
            actions = self.core.predict_actions(batch=batch)
            return LatentOutput(logits=None, tokens=None, vector=vec, actions=actions, meta=None)

        raise NotImplementedError(f"{type(self).__name__} does not support mode={mode.value!r}")
