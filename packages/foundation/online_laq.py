"""
Online LAQ label generation helpers.

These utilities adapt temporal frame tensors to the LAQ (Stage 1) encoder
to generate discrete latent action codes during Stage 2 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from common.batch_utils import temporal_frames_to_bcthw

if TYPE_CHECKING:
    from laq.inference import LAQEncoderVQInference


def frames_to_laq_video(frames: torch.Tensor) -> torch.Tensor:
    """
    Convert temporal frame tensors to LAQ input layout.

    LAQ expects:
      - video: [B, 3, T, H, W] float32 in [0, 1]
    """

    video = temporal_frames_to_bcthw(frames)

    if video.dtype == torch.uint8:
        video = video.to(torch.float32) / 255.0
    else:
        video = video.to(torch.float32)

    return video


class LatentCodeProvider(Protocol):
    codebook_size: int
    code_seq_len: int
    codebook_dim: int
    image_size: tuple[int, int]

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

    @property
    def image_size(self) -> tuple[int, int]:
        return self._encoder_vq.image_size

    @torch.no_grad()
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        return self._encoder_vq.codes_from_video(video)

    @torch.no_grad()
    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._encoder_vq.vectors_from_codes(codes)

    @torch.no_grad()
    def codes_and_vectors_from_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encoder_vq.codes_and_vectors_from_video(video)
