"""
Online LAQ label generation helpers.

These utilities adapt OpenX/OXE frame-pair batches to the LAQ (Stage 1) encoder
to generate discrete latent action codes during Stage 2 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Protocol

import torch

if TYPE_CHECKING:
    from laq.inference import LAQEncoderVQInference


def oxe_frames_to_laq_video(frames: torch.Tensor) -> torch.Tensor:
    """
    Convert OXE batch frames to LAQ input layout.

    Expected OXE layout (from `common.data.oxe_collate_fn`):
      - frames: [B, T, H, W, 3] uint8 (T can be 2 for frame pairs; future LAQ may use T>2)

    LAQ expects:
      - video: [B, 3, T, H, W] float32 in [0, 1]
    """

    if frames.ndim != 5:
        raise ValueError(f"Expected 5D frames tensor, got shape {tuple(frames.shape)}")

    # Accept either [B, T, H, W, 3] or [B, T, 3, H, W] or [B, 3, T, H, W].
    if frames.shape[-1] == 3:
        # [B, T, H, W, 3] -> [B, T, 3, H, W]
        video = frames.permute(0, 1, 4, 2, 3)
        # [B, T, 3, H, W] -> [B, 3, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
    elif frames.shape[2] == 3:
        # [B, T, 3, H, W] -> [B, 3, T, H, W]
        video = frames.permute(0, 2, 1, 3, 4)
    elif frames.shape[1] == 3:
        video = frames
    else:
        raise ValueError(
            "Unrecognized frames layout; expected last dim=3 (BHWC), or shape[2]=3 (BTCHW), or shape[1]=3 (BCTHW). "
            f"Got {tuple(frames.shape)}"
        )

    if video.dtype == torch.uint8:
        video = video.to(torch.float32) / 255.0
    else:
        video = video.to(torch.float32)

    return video


class LatentCodeProvider(Protocol):
    codebook_size: int
    code_seq_len: int
    codebook_dim: int

    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Return codebook indices [B, code_seq_len] for a video batch."""

    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Return codebook vectors [B, code_seq_len, codebook_dim] for LAQ code ids."""

    def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (code ids, codebook vectors)."""


@dataclass
class OnlineLAQConfig:
    laq_checkpoint_path: str


class LAQTaskCodeProvider(torch.nn.Module):
    """
    Thin adapter over LAQEncoderVQInference exposing code indices for Stage 2.
    """

    def __init__(self, encoder_vq: LAQEncoderVQInference):
        super().__init__()
        self._encoder_vq = encoder_vq
        self.codebook_size = encoder_vq.codebook_size
        self.code_seq_len = encoder_vq.code_seq_len
        self.codebook_dim = encoder_vq.codebook_dim

    def train(self, mode: bool = True) -> "LAQTaskCodeProvider":
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return self._encoder_vq.device

    @torch.no_grad()
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        return self._encoder_vq.codes_from_video(video)

    @torch.no_grad()
    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._encoder_vq.vectors_from_codes(codes)

    @torch.no_grad()
    def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encoder_vq.codes_and_vectors_from_video(video)


def extract_oxe_language(batch: Dict[str, Any]) -> list[str]:
    language = batch.get("language")
    if language is None:
        raise KeyError("Expected OXE batch to include 'language' (list[str])")
    if not isinstance(language, list):
        raise TypeError(f"Expected 'language' to be a list[str], got {type(language)}")
    return [str(x) for x in language]


def extract_oxe_actions(batch: Dict[str, Any]) -> torch.Tensor:
    actions = batch.get("action")
    if actions is None:
        raise KeyError("Expected OXE batch to include 'action' metadata")

    if isinstance(actions, torch.Tensor):
        out = actions.to(torch.float32)
    elif isinstance(actions, list):
        if not actions:
            raise ValueError("Expected non-empty 'action' list in OXE batch")
        first = actions[0]
        if isinstance(first, torch.Tensor):
            out = torch.stack([x.to(torch.float32) for x in actions], dim=0)
        else:
            out = torch.as_tensor(actions, dtype=torch.float32)
    else:
        out = torch.as_tensor(actions, dtype=torch.float32)

    if out.ndim != 2:
        raise ValueError(f"Expected action tensor [B, A], got {tuple(out.shape)}")
    return out


def extract_oxe_initial_state(batch: Dict[str, Any]) -> torch.Tensor | None:
    state = batch.get("initial_state")
    if state is None:
        return None

    if torch.is_tensor(state):
        out = state.to(torch.float32)
    elif isinstance(state, list):
        if not state:
            return None
        first = state[0]
        if torch.is_tensor(first):
            out = torch.stack([x.to(torch.float32) for x in state], dim=0)
        else:
            out = torch.as_tensor(state, dtype=torch.float32)
    else:
        out = torch.as_tensor(state, dtype=torch.float32)

    if out.ndim != 2:
        raise ValueError(f"Expected initial_state tensor [B, S], got {tuple(out.shape)}")
    return out
