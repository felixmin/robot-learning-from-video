from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class FoundationBatch:
    """
    Shared batch schema for Stage 2 and Stage 3.

    Canonical fields:
    - `image_streams`: mapping camera_key -> image tensor.
    - `task_text` or pretokenized language fields.
    - `state`.
    - optional supervision targets (`target_latent_vectors`, `target_actions`)
      and optional action padding mask (`action_is_pad`).
    """

    # Canonical multimodal inputs.
    image_streams: dict[str, torch.Tensor] | None = None
    image_padding_masks: dict[str, torch.Tensor] | None = None
    task_text: Sequence[str] | None = None
    subtask_text: Sequence[str] | None = None
    language_tokens: torch.Tensor | None = None  # [B, L] long
    language_attention_mask: torch.Tensor | None = None  # [B, L] bool

    # Optional supervision and state.
    target_codes: torch.Tensor | None = None  # [B, S] long (optional)
    target_latent_vectors: torch.Tensor | None = None  # [B, S, D] or [B, D] (optional)
    target_actions: torch.Tensor | None = None  # [B, A] or [B, T, A] float (optional)
    action_is_pad: torch.Tensor | None = None  # [B, T] bool (optional)
    state: torch.Tensor | None = None  # [B, S_state] or [B, T, S_state] (optional)
    meta: dict[str, Any] | None = None


class BackendMode(str, Enum):
    CODES = "codes"
    LATENT_FLOW = "latent_flow"
    ACTIONS = "actions"
    MULTITASK = "multitask"


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class LatentOutput:
    # Discrete (LAQ codes)
    logits: torch.Tensor | None = None  # [B, S, K]
    tokens: torch.Tensor | None = None  # [B, S] long
    # Continuous (future)
    vector: torch.Tensor | None = None  # [B, D] or [B, S, D]
    actions: torch.Tensor | None = None  # [B, A]
    # Small debug objects
    meta: dict[str, Any] | None = None


class VLABackend(Protocol):
    """
    Backend interface (Stage 2 + optional Stage 3).

    All model- and prompting-specific behavior lives behind this interface.
    """

    codebook_size: int
    code_seq_len: int

    def setup(self, *, device: torch.device) -> None:
        """Finalize tokenizer edits, resize embeddings, move modules to device, etc."""

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        """Compute training loss for the requested mode."""

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        """Return a latent representation for the requested mode."""
