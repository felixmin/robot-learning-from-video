"""LAM encoder+VQ inference wrapper for Stage-2 code generation."""

from __future__ import annotations

import torch

from lam.models.forward_core import normalize_video_input
from lam.models.latent_action_quantization import LatentActionQuantization

# Decoder/teacher attributes not needed for encoder+VQ inference.
# Setting these to None on the wrapped model frees their VRAM.
_DECODER_ATTRS = (
    "dino_decoder",
    "pixel_decoder",
    "pixel_to_pixels",
    "aux_decoder",
    "aux_to_pixels",
    "flow_decoder",
    "flow_teacher",
    "decoder_context_projection",
)


class LAMEncoderVQInference(torch.nn.Module):
    """
    Frozen encoder+VQ wrapper for Stage-2 code generation.

    Wraps a LatentActionQuantization model and prunes decoder/teacher
    submodules on construction to reclaim VRAM.
    """

    codebook_size: int
    code_seq_len: int
    codebook_dim: int

    def __init__(self, model: LatentActionQuantization, prune_decoders: bool = True):
        super().__init__()

        self.codebook_size = model.vq.num_embeddings
        self.code_seq_len = model.code_seq_len
        self.codebook_dim = model.vq.embedding_dim

        if prune_decoders:
            for attr in _DECODER_ATTRS:
                if getattr(model, attr, None) is not None:
                    setattr(model, attr, None)

        # Stage-2 uses this as a frozen label generator.
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._model = model

    def train(self, mode: bool = True) -> "LAMEncoderVQInference":
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(int(x) for x in self._model.image_size)

    @torch.no_grad()
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Return codebook indices [B, code_seq_len] for a video batch."""
        video = video.to(self.device)
        video = normalize_video_input(self._model, video)
        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        _, _, first_tokens, last_tokens = self._model._encode_frames(
            first_frame, rest_frames
        )
        return self._model.vq.get_indices(first_tokens, last_tokens)

    @torch.no_grad()
    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Return codebook vectors [B, code_seq_len, codebook_dim] for code ids."""
        if codes.ndim != 2:
            raise ValueError(f"Expected codes [B, S], got {tuple(codes.shape)}")
        codes = codes.to(device=self._model.vq.codebooks.device, dtype=torch.long)
        return self._model.vq.codebooks[codes]

    @torch.no_grad()
    def codes_and_vectors_from_video(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (code ids [B, S], codebook vectors [B, S, D])."""
        codes = self.codes_from_video(video)
        vectors = self.vectors_from_codes(codes)
        return codes, vectors
