from __future__ import annotations

import torch

from common.lerobot_v3_parity import compare_legacy_and_lerobot_pair_samples
from common.lerobot_v3_parity import image_stream_metrics
from common.lerobot_v3_parity import normalize_text
from common.lerobot_v3_parity import reduce_action_chunk_sum
from common.lerobot_v3_parity import summarize_parity_reports
from common.lerobot_v3_types import DatasetSample


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(" pick   up \n object ") == "pick up object"
    assert normalize_text(None) == ""


def test_reduce_action_chunk_sum_respects_pad_mask() -> None:
    action = torch.tensor([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    is_pad = torch.tensor([False, True, False])
    out = reduce_action_chunk_sum(action, is_pad)
    assert torch.equal(out, torch.tensor([101.0, 202.0]))


def test_image_stream_metrics_exact_match() -> None:
    image = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    metrics = image_stream_metrics(image, image.clone())
    assert metrics["exact_match"] is True
    assert metrics["mean_abs_diff"] == 0.0
    assert metrics["max_abs_diff"] == 0.0


def test_compare_legacy_and_lerobot_pair_samples_reports_zero_diff_for_match() -> None:
    legacy_frames = (
        torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8)
        .reshape(2, 3, 4, 4)
        .permute(1, 0, 2, 3)
    )
    legacy = {
        "frames": legacy_frames,
        "language": "pick up cube",
        "frame_idx": 7,
        "initial_state": torch.tensor([0.1, 0.2]),
        "action": torch.tensor([1.0, 5.0]),
    }
    lerobot = DatasetSample(
        image_streams={"primary": legacy_frames.permute(1, 0, 2, 3).contiguous()},
        image_padding_masks={"primary": torch.tensor([True, True])},
        state=torch.tensor([[0.1, 0.2]]),
        state_is_pad=torch.tensor([False]),
        action=torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 3.0]]),
        action_is_pad=torch.tensor([False, True, False]),
        task_text="pick up cube",
        meta={"frame_idx": 7},
    )

    report = compare_legacy_and_lerobot_pair_samples(
        legacy, lerobot, camera_role="primary"
    )

    assert report["language_exact_match"] is True
    assert report["image_mask_exact_match"] is True
    assert report["image_metrics"]["exact_match"] is True
    assert report["state_metrics"]["l2"] == 0.0
    assert report["action_metrics"]["l2"] == 0.0
    assert report["legacy_frame_idx"] == 7
    assert report["lerobot_frame_idx"] == 7


def test_summarize_parity_reports_aggregates_counts_and_means() -> None:
    reports = [
        {
            "language_exact_match": True,
            "image_mask_exact_match": True,
            "image_metrics": {"mean_abs_diff": 0.0, "max_abs_diff": 0.0},
            "state_metrics": {"l2": 0.0},
            "action_metrics": {"l2": 1.0},
        },
        {
            "language_exact_match": False,
            "image_mask_exact_match": True,
            "image_metrics": {"mean_abs_diff": 2.0, "max_abs_diff": 4.0},
            "state_metrics": {"l2": 2.0},
            "action_metrics": {"l2": 3.0},
        },
    ]

    summary = summarize_parity_reports(reports)

    assert summary["num_reports"] == 2
    assert summary["language_exact_matches"] == 1
    assert summary["image_mask_exact_matches"] == 2
    assert summary["mean_image_mean_abs_diff"] == 1.0
    assert summary["mean_image_max_abs_diff"] == 2.0
    assert summary["mean_state_l2"] == 1.0
    assert summary["mean_action_l2"] == 2.0
