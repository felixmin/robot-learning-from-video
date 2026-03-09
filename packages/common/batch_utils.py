from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch


def select_primary_image_stream(
    image_streams: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if not image_streams:
        raise ValueError("Expected at least one image stream")
    if "primary" in image_streams:
        return image_streams["primary"]
    return image_streams[next(iter(image_streams))]


def temporal_frames_to_bcthw(
    frames: torch.Tensor,
    *,
    expected_time_steps: int | None = None,
) -> torch.Tensor:
    if frames.ndim != 5:
        raise ValueError(
            f"Expected 5D temporal frames tensor, got {tuple(frames.shape)}"
        )
    if int(frames.shape[-1]) == 3:
        video = frames.permute(0, 4, 1, 2, 3).contiguous()
    elif int(frames.shape[2]) == 3:
        video = frames.permute(0, 2, 1, 3, 4).contiguous()
    elif int(frames.shape[1]) == 3:
        video = frames.contiguous()
    else:
        raise ValueError(f"Unsupported temporal frame layout: {tuple(frames.shape)}")
    if expected_time_steps is not None and int(video.shape[2]) != int(
        expected_time_steps
    ):
        raise ValueError(
            f"Expected T={int(expected_time_steps)}, got T={int(video.shape[2])}"
        )
    return video


def move_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_value_to_device(item, device) for key, item in value.items()}
    return value


def move_dataclass_tensors_to_device(batch: Any, device: torch.device) -> Any:
    return type(batch)(
        **{
            field.name: move_value_to_device(getattr(batch, field.name), device)
            for field in fields(batch)
        }
    )


def uint8_image_streams_to_float32(
    image_streams: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.to(dtype=torch.float32).div_(255.0)
            if value.dtype == torch.uint8
            else value
        )
        for key, value in image_streams.items()
    }
