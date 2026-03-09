from __future__ import annotations

from collections import Counter

import numpy as np

from common.lerobot_v3_sampler import (
    DistributedWeightedLeRobotTokenSampler,
    WeightedLeRobotTokenSampler,
)
from common.lerobot_v3_source import compile_source_index

from tests.common.lerobot_v3_fixtures import make_test_meta, make_test_request


def _build_source(*, start: int, lengths: list[int], request=None):
    episodes = []
    cursor = int(start)
    for episode_index, length in enumerate(lengths):
        episodes.append(
            {
                "episode_index": episode_index,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + int(length),
            }
        )
        cursor += int(length)
    meta = make_test_meta(episodes=episodes, fps=10)
    return compile_source_index(
        meta=meta,
        request=request or make_test_request(image_requests={"primary": (0, 1)}),
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key=None,
    )


def test_weighted_lerobot_token_sampler_respects_configured_source_weights() -> None:
    source_a = _build_source(start=0, lengths=[50, 50])
    source_b = _build_source(start=1000, lengths=[50] * 8)

    sampler = WeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.9, 0.1], dtype=np.float64),
        num_samples=2000,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    per_source = Counter(token.source_id for token in samples)
    frac0 = per_source[0] / float(len(samples))
    frac1 = per_source[1] / float(len(samples))
    assert 0.84 <= frac0 <= 0.96
    assert 0.04 <= frac1 <= 0.16


def test_weighted_lerobot_token_sampler_rebuilds_episode_cycle_after_exhaustion() -> (
    None
):
    source = _build_source(start=0, lengths=[20, 20, 20])
    sampler = WeightedLeRobotTokenSampler(
        compiled_sources=[source],
        source_weights=np.asarray([1.0], dtype=np.float64),
        num_samples=7,
        seed=3,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    first_cycle_eps = [token.episode_id for token in samples[:3]]
    assert len(set(first_cycle_eps)) == 3
    per_episode = Counter(token.episode_id for token in samples)
    assert max(per_episode.values()) - min(per_episode.values()) <= 1


def test_weighted_lerobot_token_sampler_is_deterministic_for_same_seed_and_epoch() -> (
    None
):
    source_a = _build_source(start=0, lengths=[12, 12, 12])
    source_b = _build_source(start=100, lengths=[12, 12])
    sampler_a = WeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.6, 0.4], dtype=np.float64),
        num_samples=40,
        seed=99,
        epoch=0,
        resample_each_epoch=True,
    )
    sampler_b = WeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.6, 0.4], dtype=np.float64),
        num_samples=40,
        seed=99,
        epoch=0,
        resample_each_epoch=True,
    )
    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_weighted_lerobot_token_sampler_set_epoch_changes_sequence() -> None:
    source_a = _build_source(start=0, lengths=[12, 12, 12])
    source_b = _build_source(start=100, lengths=[12, 12])
    sampler = WeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        num_samples=40,
        seed=21,
        epoch=0,
        resample_each_epoch=True,
    )
    sampler.set_epoch(0)
    seq0 = list(iter(sampler))
    sampler.set_epoch(1)
    seq1 = list(iter(sampler))
    assert seq0 != seq1


def test_distributed_weighted_lerobot_token_sampler_produces_equal_rank_lengths() -> (
    None
):
    source_a = _build_source(start=0, lengths=[20, 20, 20])
    source_b = _build_source(start=100, lengths=[20, 20, 20])

    rank0 = DistributedWeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        global_num_samples=24,
        rank=0,
        world_size=4,
        seed=7,
        epoch=0,
        resample_each_epoch=True,
    )
    rank1 = DistributedWeightedLeRobotTokenSampler(
        compiled_sources=[source_a, source_b],
        source_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        global_num_samples=24,
        rank=1,
        world_size=4,
        seed=7,
        epoch=0,
        resample_each_epoch=True,
    )
    assert len(rank0) == len(rank1) == 6


def test_distributed_weighted_lerobot_token_sampler_has_no_cross_rank_token_overlap_for_epoch_plan() -> (
    None
):
    source = _build_source(start=0, lengths=[50, 50, 50])

    rank0 = DistributedWeightedLeRobotTokenSampler(
        compiled_sources=[source],
        source_weights=np.asarray([1.0], dtype=np.float64),
        global_num_samples=12,
        rank=0,
        world_size=3,
        seed=11,
        epoch=0,
        resample_each_epoch=True,
    )
    rank1 = DistributedWeightedLeRobotTokenSampler(
        compiled_sources=[source],
        source_weights=np.asarray([1.0], dtype=np.float64),
        global_num_samples=12,
        rank=1,
        world_size=3,
        seed=11,
        epoch=0,
        resample_each_epoch=True,
    )
    rank2 = DistributedWeightedLeRobotTokenSampler(
        compiled_sources=[source],
        source_weights=np.asarray([1.0], dtype=np.float64),
        global_num_samples=12,
        rank=2,
        world_size=3,
        seed=11,
        epoch=0,
        resample_each_epoch=True,
    )

    set0 = set(iter(rank0))
    set1 = set(iter(rank1))
    set2 = set(iter(rank2))
    assert set0.isdisjoint(set1)
    assert set0.isdisjoint(set2)
    assert set1.isdisjoint(set2)
