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


def test_sampler_no_replacement_when_pairs_per_episode_is_one() -> None:
    index = _build_index(
        steps_by_dataset=[[6, 5, 4], [8, 7]],
        dataset_weights=[0.95, 0.05],
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

    samples = list(iter(sampler))
    assert len(samples) == len(sampler) == 5
    assert len(set(samples)) == len(samples)

    per_episode = Counter(ep for ep, _ in samples)
    assert set(per_episode.values()) == {1}


def test_sampler_redistributes_when_dataset_quota_exceeds_capacity() -> None:
    index = _build_index(
        steps_by_dataset=[[5, 5], [5, 5, 5, 5, 5, 5, 5, 5]],
        dataset_weights=[0.99, 0.01],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=10,
        seed=7,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    assert len(samples) == 10
    assert len(set(samples)) == 10

    dataset_ids = index.episode_dataset_ids
    per_dataset = Counter(int(dataset_ids[ep]) for ep, _ in samples)
    assert per_dataset[0] == 2
    assert per_dataset[1] == 8


def test_sampler_no_replacement_for_timesteps_within_episode() -> None:
    index = _build_index(
        steps_by_dataset=[[4]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=None,
        weights_by_size=False,
        num_samples=4,
        seed=11,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    assert len(samples) == 4
    assert len(set(samples)) == 4
    assert {t for _, t in samples} == {0, 1, 2, 3}


def test_sampler_falls_back_to_replacement_only_after_unique_capacity() -> None:
    index = _build_index(
        steps_by_dataset=[[10, 10, 10]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=7,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
    )

    samples = list(iter(sampler))
    assert len(samples) == 7
    assert len(set(samples)) < len(samples)
    assert len(set(samples)) <= 3


def test_sampler_fresh_per_draw_varies_t_on_repeated_episode_draws() -> None:
    index = _build_index(
        steps_by_dataset=[[10]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=6,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
        repeat_t_policy="fresh_per_draw",
    )

    samples = list(iter(sampler))
    assert len(samples) == 6
    assert all(ep == 0 for ep, _ in samples)

    t_values = [int(t) for _, t in samples]
    assert len(set(t_values)) > 1


def test_sampler_fresh_per_draw_uses_weighted_replacement_without_capacity_collapse() -> None:
    index = _build_index(
        steps_by_dataset=[[10, 10], [10, 10, 10, 10, 10, 10, 10, 10]],
        dataset_weights=[0.99, 0.01],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=1000,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
        repeat_t_policy="fresh_per_draw",
    )

    samples = list(iter(sampler))
    assert len(samples) == 1000

    dataset_ids = index.episode_dataset_ids
    per_dataset = Counter(int(dataset_ids[ep]) for ep, _ in samples)
    # Replacement mode should follow configured weights instead of clipping to
    # unique episode capacity (which would over-sample dataset 1 here).
    assert per_dataset[0] > 900
    assert per_dataset[1] < 100


def test_sampler_cached_subset_keeps_t_fixed_on_repeated_episode_draws() -> None:
    index = _build_index(
        steps_by_dataset=[[10]],
        dataset_weights=[1.0],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=6,
        seed=5,
        epoch=0,
        resample_each_epoch=True,
        repeat_t_policy="cached_subset",
    )

    samples = list(iter(sampler))
    assert len(samples) == 6
    assert all(ep == 0 for ep, _ in samples)

    t_values = [int(t) for _, t in samples]
    assert len(set(t_values)) == 1


def test_all_exhausted_epoch_length_matches_total_capacity() -> None:
    index = _build_index(
        steps_by_dataset=[[100] * 100, [1000] * 1000],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=None,
        seed=0,
        epoch=0,
        resample_each_epoch=True,
        stopping_strategy="all_exhausted",
    )

    assert len(sampler) == 1100


def test_first_exhausted_epoch_length_uses_min_capacity_over_probability() -> None:
    index = _build_index(
        steps_by_dataset=[[100], [1000]],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=None,
        seed=0,
        epoch=0,
        resample_each_epoch=True,
        stopping_strategy="first_exhausted",
    )

    # capacities are [1, 1] because pairs_per_episode=1 and one episode per dataset.
    # with probabilities [0.5, 0.5], epoch size is floor(min(capacity_i / p_i)) = 2.
    assert len(sampler) == 2


def test_first_exhausted_epoch_length_balances_before_small_dataset_exhausts() -> None:
    index = _build_index(
        steps_by_dataset=[[100] * 100, [1000] * 1000],
        dataset_weights=[0.5, 0.5],
    )
    sampler = OpenXLocalIndexedEpisodePairSampler(
        index=index,
        pairs_per_episode=1,
        weights_by_size=False,
        num_samples=None,
        seed=0,
        epoch=0,
        resample_each_epoch=True,
        stopping_strategy="first_exhausted",
    )
    assert len(sampler) == 200


def test_sampler_rejects_unknown_stopping_strategy() -> None:
    index = _build_index(
        steps_by_dataset=[[4]],
        dataset_weights=[1.0],
    )
    try:
        OpenXLocalIndexedEpisodePairSampler(
            index=index,
            pairs_per_episode=1,
            weights_by_size=False,
            num_samples=None,
            seed=0,
            epoch=0,
            resample_each_epoch=True,
            stopping_strategy="bad_value",
        )
    except ValueError as exc:
        assert "stopping_strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown stopping_strategy")


def test_sampler_rejects_unknown_repeat_t_policy() -> None:
    index = _build_index(
        steps_by_dataset=[[4]],
        dataset_weights=[1.0],
    )
    try:
        OpenXLocalIndexedEpisodePairSampler(
            index=index,
            pairs_per_episode=1,
            weights_by_size=False,
            num_samples=None,
            seed=0,
            epoch=0,
            resample_each_epoch=True,
            repeat_t_policy="bad_policy",
        )
    except ValueError as exc:
        assert "repeat_t_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown repeat_t_policy")
