from __future__ import annotations

import pytest
import torch

from stage2.online_lam import LAMTaskCodeProvider, frames_to_lam_video


def test_frames_to_lam_video_from_b2hwc3_uint8():
    frames = torch.randint(0, 256, (2, 2, 32, 32, 3), dtype=torch.uint8)
    video = frames_to_lam_video(frames)
    assert video.shape == (2, 3, 2, 32, 32)
    assert video.dtype == torch.float32
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0


def test_frames_to_lam_video_from_b23hw_uint8():
    frames = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    video = frames_to_lam_video(frames)
    assert video.shape == (2, 3, 2, 16, 16)
    assert video.dtype == torch.float32


def test_frames_to_lam_video_from_b32hw_float():
    frames = torch.rand((2, 3, 2, 8, 8), dtype=torch.float32)
    video = frames_to_lam_video(frames)
    assert video.shape == (2, 3, 2, 8, 8)
    assert video.dtype == torch.float32


def test_frames_to_lam_video_rejects_bad_shape():
    with pytest.raises(ValueError):
        frames_to_lam_video(torch.zeros((2, 3, 4)))


class _FakeEncoderVQ(torch.nn.Module):
    codebook_size = 3
    code_seq_len = 2
    codebook_dim = 2

    def __init__(self):
        super().__init__()
        self._codebooks = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @property
    def device(self) -> torch.device:
        return self._codebooks.device

    @torch.no_grad()
    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        batch_size = int(video.shape[0])
        return torch.tensor(
            [[0, 2]] * batch_size, dtype=torch.long, device=video.device
        )

    @torch.no_grad()
    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._codebooks[codes.to(self._codebooks.device)]

    @torch.no_grad()
    def codes_and_vectors_from_video(self, video: torch.Tensor):
        codes = self.codes_from_video(video)
        return codes, self.vectors_from_codes(codes)


def test_lam_provider_codes_and_vectors():
    provider = LAMTaskCodeProvider(_FakeEncoderVQ())
    video = torch.rand((2, 3, 2, 8, 8), dtype=torch.float32)

    codes, vectors = provider.codes_and_vectors_from_video(video)
    assert codes.shape == (2, 2)
    assert vectors.shape == (2, 2, 2)
    assert torch.allclose(vectors[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(vectors[0, 1], torch.tensor([1.0, 1.0]))
