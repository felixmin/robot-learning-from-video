from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def resize_with_pad(
    img: torch.Tensor, width: int, height: int, pad_value: float = -1.0
) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got shape {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def gpu_preprocess_images(
    frames: torch.Tensor,
    target_size: tuple[int, int] = (384, 384),
    normalize: bool = True,
) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        frames = frames.float()

    img = resize_with_pad(frames, target_size[1], target_size[0], pad_value=0.0)

    if normalize:
        img = img * 2.0 - 1.0

    return img


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)


def infer_hidden_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise AttributeError("Could not infer hidden size from model.config")


def get_last_layer_module(
    model: Any, optimized: bool = False
) -> torch.nn.Module | None:
    if optimized:
        if hasattr(model, "model") and hasattr(model.model, "text_model"):
            text_model = model.model.text_model
            if hasattr(text_model, "layers"):
                return text_model.layers[-1]
        if hasattr(model, "layers"):
            return model.layers[-1]
    else:
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers[-1]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[-1]
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[-1]
    return None


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector
