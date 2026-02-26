from __future__ import annotations

from collections import Counter

import numpy as np

from common.adapters.openx_local_indexed_full import (
    OpenXLocalEpisodeIndexBundle,
    OpenXLocalIndexedEpisodePairSampler,
)


def _build_index(
    *,
    steps_by_dataset: list[list[int]],
    dataset_weights: list[float],
) -> OpenXLocalEpisodeIndexBundle:
    datasets = [f"ds_{i}" for i in range(len(steps_by_dataset))]
    episode_dataset_ids: list[int] = []
    episode_num_steps: list[int] = []
    for dataset_id, steps_list in enumerate(steps_by_dataset):
        for n_steps in steps_list:
            episode_dataset_ids.append(dataset_id)
            episode_num_steps.append(int(n_steps))

    n_episodes = len(episode_dataset_ids)
    return OpenXLocalEpisodeIndexBundle(
        index_path="",
        key="test",
        datasets=datasets,
        dataset_weights=np.asarray(dataset_weights, dtype=np.float64),
        dataset_offsets=np.zeros((len(datasets),), dtype=np.int64),
        dataset_configs=[],
        shard_paths=[""] * max(1, n_episodes),
        episode_dataset_ids=np.asarray(episode_dataset_ids, dtype=np.int64),
        episode_shard_ids=np.zeros((n_episodes,), dtype=np.int64),
        episode_offsets=np.zeros((n_episodes,), dtype=np.uint64),
        episode_sizes=np.zeros((n_episodes,), dtype=np.uint64),
        episode_num_steps=np.asarray(episode_num_steps, dtype=np.int64),
        metadata={},
    )


def test_num_samples_defaults_to_total_episode_slots() -> None:
    index = _build_index(
        steps_by_dataset=[[6, 5, 4], [8, 7]],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=None,
        seed=123,
        epoch=0,
        resample_each_epoch=True,
    )
    assert len(sampler) == 5


def test_pairs_per_episode_scales_cycle_length() -> None:
    index = _build_index(
        steps_by_dataset=[[10, 10, 10]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=2,
        weights_by_size=False,
        num_samples=None,
        seed=7,
        epoch=0,
        resample_each_epoch=True,
    )
    assert len(sampler) == 6

    samples = list(iter(sampler))
    per_episode = Counter(ep for ep, _ in samples)
    assert sorted(per_episode.values()) == [2, 2, 2]


def test_cycle_rebuild_reuses_episodes_after_full_pass() -> None:
    index = _build_index(
        steps_by_dataset=[[20, 20, 20]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=7,
        seed=3,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    assert len(samples) == 7

    first_cycle_eps = [ep for ep, _ in samples[:3]]
    assert len(set(first_cycle_eps)) == 3

    per_episode = Counter(ep for ep, _ in samples)
    assert max(per_episode.values()) - min(per_episode.values()) <= 1


def test_weighted_dataset_sampling_follows_configured_weights() -> None:
    index = _build_index(
        steps_by_dataset=[[10, 10], [10] * 8],
        dataset_weights=[0.9, 0.1],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=2000,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    dataset_ids = index.episode_dataset_ids
    per_dataset = Counter(int(dataset_ids[ep]) for ep, _ in samples)
    frac0 = per_dataset[0] / float(len(samples))
    frac1 = per_dataset[1] / float(len(samples))
    assert 0.84 <= frac0 <= 0.96
    assert 0.04 <= frac1 <= 0.16


def test_repeated_episode_draws_vary_timesteps() -> None:
    index = _build_index(
        steps_by_dataset=[[100]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=24,
        seed=11,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    assert all(ep == 0 for ep, _ in samples)
    t_values = [int(t) for _, t in samples]
    assert len(set(t_values)) >= 12


def test_sampler_is_deterministic_for_same_seed_and_epoch() -> None:
    index = _build_index(
        steps_by_dataset=[[12, 12, 12], [12, 12]],
        dataset_weights=[0.6, 0.4],
    )
    sampler_a = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=40,
        seed=99,
        epoch=0,
        resample_each_epoch=True,
    )
    sampler_b = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=40,
        seed=99,
        epoch=0,
        resample_each_epoch=True,
    )
    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_set_epoch_changes_sequence_when_resampling_enabled() -> None:
    index = _build_index(
        steps_by_dataset=[[12, 12, 12], [12, 12]],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
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


def test_set_epoch_does_not_change_sequence_when_resampling_disabled() -> None:
    index = _build_index(
        steps_by_dataset=[[12, 12, 12], [12, 12]],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=40,
        seed=21,
        epoch=0,
        resample_each_epoch=False,
    )
    sampler.set_epoch(0)
    seq0 = list(iter(sampler))
    sampler.set_epoch(1)
    seq1 = list(iter(sampler))
    assert seq0 == seq1
