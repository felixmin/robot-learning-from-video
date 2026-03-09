from __future__ import annotations

import numpy as np

from common.lerobot_v3_source import compile_source_index

from tests.helpers.lerobot_v3_fixtures import make_test_meta, make_test_request


def test_compile_source_index_computes_valid_anchor_ranges_for_strict_pair_request() -> (
    None
):
    meta = make_test_meta(
        episodes=[
            {"episode_index": 0, "dataset_from_index": 100, "dataset_to_index": 110}
        ],
        fps=10,
    )
    request = make_test_request(
        image_requests={"primary": (-2, 0, 3)}, pad_missing_future=True
    )

    out = compile_source_index(
        meta=meta,
        request=request,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key=None,
    )

    assert np.array_equal(
        out.episodes.valid_anchor_start, np.asarray([102], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_end, np.asarray([107], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_count, np.asarray([5], dtype=np.int32)
    )


def test_compile_source_index_allows_full_episode_range_when_future_padding_enabled() -> (
    None
):
    meta = make_test_meta(
        episodes=[
            {"episode_index": 0, "dataset_from_index": 20, "dataset_to_index": 30}
        ],
        fps=10,
    )
    request = make_test_request(
        image_requests={"primary": (0,)},
        action_deltas=(0, 1, 2),
        pad_missing_future=True,
    )

    out = compile_source_index(
        meta=meta,
        request=request,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key="action",
    )

    assert np.array_equal(
        out.episodes.valid_anchor_start, np.asarray([20], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_end, np.asarray([30], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_count, np.asarray([10], dtype=np.int32)
    )


def test_compile_source_index_drops_unsampleable_episodes() -> None:
    meta = make_test_meta(
        episodes=[
            {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 2},
            {"episode_index": 1, "dataset_from_index": 2, "dataset_to_index": 10},
        ],
        fps=10,
    )
    request = make_test_request(
        image_requests={"primary": (0, 5)}, pad_missing_future=True
    )

    out = compile_source_index(
        meta=meta,
        request=request,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key=None,
    )

    assert np.array_equal(out.sampleable_episode_ids, np.asarray([1], dtype=np.int32))
    assert np.array_equal(
        out.episodes.valid_anchor_count, np.asarray([0, 3], dtype=np.int32)
    )


def test_compile_source_index_uses_absolute_dataset_indices() -> None:
    meta = make_test_meta(
        episodes=[
            {"episode_index": 7, "dataset_from_index": 50, "dataset_to_index": 55}
        ],
        fps=10,
    )
    request = make_test_request(image_requests={"primary": (0,)})

    out = compile_source_index(
        meta=meta,
        request=request,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key=None,
    )

    assert np.array_equal(
        out.episodes.dataset_from_index, np.asarray([50], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.dataset_to_index, np.asarray([55], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_start, np.asarray([50], dtype=np.int64)
    )
    assert np.array_equal(
        out.episodes.valid_anchor_end, np.asarray([55], dtype=np.int64)
    )
