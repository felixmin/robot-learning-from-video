from __future__ import annotations

import torch

from foundation.backends.interfaces import FoundationBatch
from foundation.backends.smolvla_shared.model import SmolVLASharedCore


def latent_flow_loss(
    *,
    core: SmolVLASharedCore,
    batch: FoundationBatch,
    target_vectors: torch.Tensor,
) -> torch.Tensor:
    return core.latent_flow_loss(batch=batch, target_vectors=target_vectors)


def action_flow_loss(
    *,
    core: SmolVLASharedCore,
    batch: FoundationBatch,
    target_actions: torch.Tensor,
    action_is_pad: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    return core.action_flow_loss(
        batch=batch,
        target_actions=target_actions,
        action_is_pad=action_is_pad,
        reduction=reduction,
    )
