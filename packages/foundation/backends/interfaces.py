from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class FoundationBatch:
    """
    Minimal common batch shape for Stage 2.

    - `frames` is expected to include the frame sequence used by the LAQ teacher
      (today typically a pair t,t+Δ; future may use T>2).
    - Device expectation: frames live on CPU (uint8) because many processors
      require PIL/NumPy conversion.
    """

    frames: torch.Tensor
    instructions: Sequence[str]
    target_codes: torch.Tensor | None = None  # [B, S] long (optional)
    target_latent_vectors: torch.Tensor | None = None  # [B, S, D] or [B, D] (optional)
    target_actions: torch.Tensor | None = None  # [B, A] float (optional)
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
