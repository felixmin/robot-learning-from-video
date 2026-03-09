from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from common.lerobot_v3_types import DatasetSample


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def reduce_action_chunk_sum(
    action: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor | None:
    if action is None:
        return None
    action_t = torch.as_tensor(action, dtype=torch.float32)
    if action_t.ndim == 1:
        return action_t
    if action_t.ndim != 2:
        raise ValueError(
            f"Expected action tensor rank 1 or 2, got {tuple(action_t.shape)}"
        )
    if action_is_pad is None:
        valid = torch.ones((int(action_t.shape[0]),), dtype=torch.bool)
    else:
        valid = ~torch.as_tensor(action_is_pad, dtype=torch.bool).reshape(-1)
        if int(valid.shape[0]) != int(action_t.shape[0]):
            raise ValueError(
                "Action pad mask length does not match action sequence length: "
                f"{int(valid.shape[0])} vs {int(action_t.shape[0])}"
            )
    if not bool(valid.any()):
        return torch.zeros((int(action_t.shape[-1]),), dtype=torch.float32)
    return action_t[valid].sum(dim=0)


def image_stream_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    ref = torch.as_tensor(reference, dtype=torch.float32)
    cand = torch.as_tensor(candidate, dtype=torch.float32)
    if tuple(ref.shape) != tuple(cand.shape):
        raise ValueError(
            f"Image shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}"
        )
    diff = (ref - cand).abs()
    mse = torch.mean((ref - cand) ** 2).item()
    return {
        "exact_match": bool(torch.equal(ref, cand)),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "mse": float(mse),
    }


def vector_metrics(
    reference: torch.Tensor | Sequence[float] | None,
    candidate: torch.Tensor | Sequence[float] | None,
) -> dict[str, float | int | bool] | None:
    if reference is None and candidate is None:
        return None
    if reference is None or candidate is None:
        return {
            "present_in_both": False,
            "reference_dim": (
                0 if reference is None else int(torch.as_tensor(reference).numel())
            ),
            "candidate_dim": (
                0 if candidate is None else int(torch.as_tensor(candidate).numel())
            ),
        }

    ref = torch.as_tensor(reference, dtype=torch.float32).reshape(-1)
    cand = torch.as_tensor(candidate, dtype=torch.float32).reshape(-1)
    shared_dim = min(int(ref.numel()), int(cand.numel()))
    if shared_dim == 0:
        return {
            "present_in_both": True,
            "reference_dim": int(ref.numel()),
            "candidate_dim": int(cand.numel()),
            "shared_dim": 0,
            "exact_match": bool(ref.numel() == cand.numel()),
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "l2": 0.0,
        }
    diff = (ref[:shared_dim] - cand[:shared_dim]).abs()
    return {
        "present_in_both": True,
        "reference_dim": int(ref.numel()),
        "candidate_dim": int(cand.numel()),
        "shared_dim": int(shared_dim),
        "exact_match": bool(torch.equal(ref, cand)),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "l2": float(
            torch.linalg.vector_norm(ref[:shared_dim] - cand[:shared_dim]).item()
        ),
    }


def _legacy_frames_to_tchw(frames: torch.Tensor) -> torch.Tensor:
    frames_t = torch.as_tensor(frames)
    if frames_t.ndim != 4:
        raise ValueError(
            f"Expected legacy pair frames rank 4, got {tuple(frames_t.shape)}"
        )
    if int(frames_t.shape[0]) == 3:
        return frames_t.permute(1, 0, 2, 3).contiguous()
    if int(frames_t.shape[-1]) == 3:
        return frames_t.permute(2, 0, 1, 3).contiguous()
    raise ValueError(f"Unsupported legacy frame layout: {tuple(frames_t.shape)}")


def compare_legacy_and_lerobot_pair_samples(
    legacy_sample: dict[str, Any],
    lerobot_sample: DatasetSample,
    *,
    camera_role: str,
) -> dict[str, Any]:
    if (
        lerobot_sample.image_streams is None
        or camera_role not in lerobot_sample.image_streams
    ):
        raise KeyError(f"Missing camera role {camera_role!r} in LeRobot sample")

    legacy_frames = _legacy_frames_to_tchw(torch.as_tensor(legacy_sample["frames"]))
    lerobot_frames = torch.as_tensor(lerobot_sample.image_streams[camera_role])
    if tuple(legacy_frames.shape) != tuple(lerobot_frames.shape):
        raise ValueError(
            "Legacy and LeRobot image shapes differ after layout normalization: "
            f"{tuple(legacy_frames.shape)} vs {tuple(lerobot_frames.shape)}"
        )

    legacy_mask = torch.ones((int(legacy_frames.shape[0]),), dtype=torch.bool)
    lerobot_mask = None
    if (
        lerobot_sample.image_padding_masks is not None
        and camera_role in lerobot_sample.image_padding_masks
    ):
        lerobot_mask = torch.as_tensor(
            lerobot_sample.image_padding_masks[camera_role], dtype=torch.bool
        )
    if lerobot_mask is None:
        lerobot_mask = torch.ones((int(lerobot_frames.shape[0]),), dtype=torch.bool)

    lerobot_state = None
    if lerobot_sample.state is not None:
        state = torch.as_tensor(lerobot_sample.state, dtype=torch.float32)
        lerobot_state = state[0] if state.ndim > 1 else state

    lerobot_action = reduce_action_chunk_sum(
        lerobot_sample.action, lerobot_sample.action_is_pad
    )

    return {
        "language_exact_match": normalize_text(legacy_sample.get("language"))
        == normalize_text(lerobot_sample.task_text),
        "image_mask_exact_match": bool(torch.equal(legacy_mask, lerobot_mask)),
        "image_metrics": image_stream_metrics(legacy_frames, lerobot_frames),
        "state_metrics": vector_metrics(
            legacy_sample.get("initial_state"), lerobot_state
        ),
        "action_metrics": vector_metrics(legacy_sample.get("action"), lerobot_action),
        "legacy_frame_idx": int(legacy_sample.get("frame_idx", -1)),
        "lerobot_frame_idx": (
            None
            if lerobot_sample.meta is None or "frame_idx" not in lerobot_sample.meta
            else int(lerobot_sample.meta["frame_idx"])
        ),
    }


def summarize_parity_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("Cannot summarize empty parity report list")

    def _mean(key: str, *, section: str) -> float | None:
        values = [
            float(report[section][key])
            for report in reports
            if report.get(section) is not None and key in report[section]
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "num_reports": int(len(reports)),
        "language_exact_matches": int(
            sum(bool(report["language_exact_match"]) for report in reports)
        ),
        "image_mask_exact_matches": int(
            sum(bool(report["image_mask_exact_match"]) for report in reports)
        ),
        "mean_image_mean_abs_diff": _mean("mean_abs_diff", section="image_metrics"),
        "mean_image_max_abs_diff": _mean("max_abs_diff", section="image_metrics"),
        "mean_state_l2": _mean("l2", section="state_metrics"),
        "mean_action_l2": _mean("l2", section="action_metrics"),
    }
