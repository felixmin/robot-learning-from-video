from __future__ import annotations

from types import SimpleNamespace

import torch

from lam.validation.visualization import BasicVisualizationStrategy


class _NoAuxModel:
    aux_decoder = None
    pixel_decoder = None

    def __call__(self, *args, **kwargs):
        raise AssertionError("Model should not be called without a reconstruction decoder")


def test_basic_visualization_falls_back_to_raw_frame_pairs_without_aux_decoder() -> (
    None
):
    strategy = BasicVisualizationStrategy()
    pl_module = SimpleNamespace(
        model=_NoAuxModel(),
        training=True,
        device=torch.device("cpu"),
    )
    frames = torch.rand(3, 3, 2, 8, 8, dtype=torch.float32)

    grid = strategy._create_recon_grid(frames, pl_module)

    assert grid is not None
    assert isinstance(grid, torch.Tensor)
    assert tuple(grid.shape)[0] == 3


class _PixelOnlyModel:
    aux_decoder = None
    pixel_decoder = object()

    def __call__(self, video, **kwargs):
        assert kwargs["return_recons_only"] is True
        batch = video.shape[0]
        height = video.shape[-2]
        width = video.shape[-1]
        return torch.full((batch, 3, height, width), 0.5, dtype=video.dtype)


def test_basic_visualization_uses_pixel_decoder_when_aux_decoder_is_disabled() -> None:
    strategy = BasicVisualizationStrategy()
    pl_module = SimpleNamespace(
        model=_PixelOnlyModel(),
        training=True,
        device=torch.device("cpu"),
        eval=lambda: None,
        train=lambda mode: None,
    )
    frames = torch.rand(3, 3, 2, 8, 8, dtype=torch.float32)

    grid = strategy._create_recon_grid(frames, pl_module)

    assert grid is not None
    assert isinstance(grid, torch.Tensor)
    assert tuple(grid.shape)[0] == 3
