from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.compute_stats import aggregate_stats

from common.lerobot_v3_stats import build_run_normalization_stats, merge_weighted_stats

from tests.common.lerobot_v3_fixtures import make_test_source_stats


def test_merge_weighted_stats_returns_single_source_stats_unchanged() -> None:
    stats = make_test_source_stats(action_mean=[1.0, 2.0], action_std=[3.0, 4.0])
    out = merge_weighted_stats([stats], np.asarray([1.0], dtype=np.float64))
    assert np.allclose(out["action"]["mean"], stats["action"]["mean"])
    assert np.allclose(out["action"]["std"], stats["action"]["std"])
    assert np.allclose(out["action"]["min"], stats["action"]["min"])
    assert np.allclose(out["action"]["max"], stats["action"]["max"])


def test_merge_weighted_stats_respects_source_weights_for_mean_and_std() -> None:
    stats_a = make_test_source_stats(
        action_mean=[0.0], action_std=[1.0], action_min=[-2.0], action_max=[2.0]
    )
    stats_b = make_test_source_stats(
        action_mean=[10.0], action_std=[1.0], action_min=[8.0], action_max=[12.0]
    )

    out = merge_weighted_stats(
        [stats_a, stats_b],
        np.asarray([0.8, 0.2], dtype=np.float64),
    )

    assert np.allclose(out["action"]["mean"], np.asarray([2.0]))
    assert np.allclose(out["action"]["std"], np.asarray([np.sqrt(17.0)]))


def test_merge_weighted_stats_keeps_global_min_and_max() -> None:
    stats_a = make_test_source_stats(
        action_mean=[0.0], action_std=[1.0], action_min=[-5.0], action_max=[1.0]
    )
    stats_b = make_test_source_stats(
        action_mean=[1.0], action_std=[2.0], action_min=[-2.0], action_max=[9.0]
    )

    out = merge_weighted_stats(
        [stats_a, stats_b],
        np.asarray([0.5, 0.5], dtype=np.float64),
    )

    assert np.allclose(out["action"]["min"], np.asarray([-5.0]))
    assert np.allclose(out["action"]["max"], np.asarray([9.0]))


def test_merge_weighted_stats_matches_lerobot_aggregate_stats_for_equal_weights_and_counts() -> (
    None
):
    stats_a = make_test_source_stats(
        action_mean=[0.0], action_std=[1.0], action_count=100
    )
    stats_b = make_test_source_stats(
        action_mean=[10.0], action_std=[3.0], action_count=100
    )

    expected = aggregate_stats([stats_a, stats_b])
    out = merge_weighted_stats(
        [stats_a, stats_b],
        np.asarray([1.0, 1.0], dtype=np.float64),
    )

    assert np.allclose(out["action"]["mean"], expected["action"]["mean"])
    assert np.allclose(out["action"]["std"], expected["action"]["std"])


def test_build_run_normalization_stats_uses_selected_sources_only() -> None:
    class _Source:
        def __init__(self, stats, weight):
            self.meta = type("Meta", (), {"stats": stats})()
            self.weight = weight

    stats_a = make_test_source_stats(action_mean=[0.0], action_std=[1.0])
    stats_b = make_test_source_stats(action_mean=[10.0], action_std=[1.0])
    stats_c = make_test_source_stats(action_mean=[100.0], action_std=[1.0])

    out = build_run_normalization_stats(
        sources=[
            _Source(stats_a, 0.75),
            _Source(stats_b, 0.25),
        ],
        weights_mode="explicit",
    )

    assert np.allclose(out["action"]["mean"], np.asarray([2.5]))
    assert not np.allclose(out["action"]["mean"], stats_c["action"]["mean"])


def test_merge_weighted_stats_handles_sources_with_different_feature_dims() -> None:
    stats_a = make_test_source_stats(
        action_mean=[0.0, 2.0], action_std=[1.0, 3.0], action_count=100
    )
    stats_b = make_test_source_stats(
        action_mean=[10.0, 20.0, 30.0], action_std=[2.0, 4.0, 6.0], action_count=100
    )

    out = merge_weighted_stats(
        [stats_a, stats_b],
        np.asarray([0.5, 0.5], dtype=np.float64),
    )

    assert np.allclose(out["action"]["mean"], np.asarray([5.0, 11.0, 30.0]))
    assert np.allclose(
        out["action"]["std"],
        np.asarray(
            [
                np.sqrt(27.5),
                np.sqrt(93.5),
                6.0,
            ]
        ),
    )


def test_build_run_normalization_stats_rejects_unsupported_weights_mode() -> None:
    class _Source:
        def __init__(self, stats, weight):
            self.meta = type("Meta", (), {"stats": stats})()
            self.weight = weight

    with pytest.raises(NotImplementedError):
        build_run_normalization_stats(
            sources=[
                _Source(
                    make_test_source_stats(action_mean=[0.0], action_std=[1.0]), 1.0
                )
            ],
            weights_mode="size_balanced",
        )
