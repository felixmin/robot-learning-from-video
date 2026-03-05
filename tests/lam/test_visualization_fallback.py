from __future__ import annotations

from types import SimpleNamespace

import torch

from lam.validation.visualization import BasicVisualizationStrategy


class _NoAuxModel:
    aux_decoder = None

    def __call__(self, *args, **kwargs):
        raise AssertionError("Model should not be called when aux_decoder is disabled")


def test_basic_visualization_falls_back_to_raw_frame_pairs_without_aux_decoder() -> None:
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
