from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from stage2.backends.interfaces import Stage2Batch
from stage2.backends.smolvla_shared.preprocess import gpu_preprocess_images, pad_vector


@dataclass(frozen=True)
class ImageTransformConfig:
    image_size: tuple[int, int]
    normalize_to_neg_one_to_one: bool
    camera_keys: Sequence[str] | None
    empty_cameras: int


@dataclass(frozen=True)
class LanguageTransformConfig:
    system_prompt: str | None
    tokenizer_max_length: int
    pad_language_to: str


def infer_batch_size(batch: Stage2Batch) -> int:
    if batch.language_tokens is not None:
        return int(batch.language_tokens.shape[0])
    if batch.image_streams:
        key = next(iter(batch.image_streams))
        return int(batch.image_streams[key].shape[0])
    if batch.state is not None:
        return int(batch.state.shape[0])
    if batch.target_latent_vectors is not None:
        return int(batch.target_latent_vectors.shape[0])
    if batch.target_actions is not None:
        return int(batch.target_actions.shape[0])
    if batch.task_text is not None:
        return len(batch.task_text)
    raise ValueError("Could not infer batch size from Stage2Batch.")


def _ensure_text_count(texts: Sequence[str], *, expected: int) -> list[str]:
    out = [str(x) for x in texts]
    if len(out) != expected:
        raise ValueError(
            f"Text batch size mismatch: expected {expected}, got {len(out)}"
        )
    return out


def _build_instruction_texts(
    *,
    instructions: Sequence[str],
    system_prompt: str | None,
) -> list[str]:
    texts: list[str] = []
    for instr in instructions:
        line = str(instr)
        if not line.endswith("\n"):
            line = f"{line}\n"
        if system_prompt:
            texts.append(f"{system_prompt}\n{line}")
        else:
            texts.append(line)
    return texts


def prepare_language_inputs(
    *,
    batch: Stage2Batch,
    tokenizer: Any,
    device: torch.device,
    config: LanguageTransformConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.language_tokens is not None or batch.language_attention_mask is not None:
        if batch.language_tokens is None or batch.language_attention_mask is None:
            raise ValueError(
                "Both language_tokens and language_attention_mask must be set together."
            )
        tokens = batch.language_tokens.to(device=device, dtype=torch.long)
        mask = batch.language_attention_mask.to(device=device, dtype=torch.bool)
        if tokens.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                "Expected language_tokens/language_attention_mask with rank 2, "
                f"got {tokens.ndim}/{mask.ndim}"
            )
        if tokens.shape != mask.shape:
            raise ValueError(
                f"language token/mask shape mismatch: {tuple(tokens.shape)} vs {tuple(mask.shape)}"
            )
        return tokens, mask

    if batch.task_text is None:
        raise ValueError(
            "Expected either pretokenized language fields or task_text in Stage2Batch."
        )
    bsize = infer_batch_size(batch)
    instructions = _ensure_text_count(batch.task_text, expected=bsize)

    texts = _build_instruction_texts(
        instructions=instructions,
        system_prompt=config.system_prompt,
    )
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=str(config.pad_language_to),
        truncation=True,
        max_length=int(config.tokenizer_max_length),
    )
    if "input_ids" not in tok or "attention_mask" not in tok:
        raise RuntimeError("tokenizer output must contain input_ids and attention_mask")
    return (
        tok["input_ids"].to(device=device, dtype=torch.long),
        tok["attention_mask"].to(device=device, dtype=torch.bool),
    )


def _select_image_step(stream: torch.Tensor, *, use_last: bool) -> torch.Tensor:
    if stream.ndim == 4:
        return stream
    if stream.ndim != 5:
        raise ValueError(
            f"Expected 4D/5D image tensor, got shape {tuple(stream.shape)}"
        )

    idx = -1 if use_last else 0
    if stream.shape[-1] == 3:
        return stream[:, idx, ...]
    if stream.shape[2] == 3:
        return stream[:, idx, ...]
    if stream.shape[1] == 3:
        return stream[:, :, idx, ...]
    raise ValueError(f"Unsupported 5D image layout: {tuple(stream.shape)}")


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {tuple(image.shape)}")
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2)
    if image.shape[1] == 3:
        return image
    raise ValueError(f"Unsupported image layout (need C=3): {tuple(image.shape)}")


def _extract_padding_mask(
    *,
    mask: torch.Tensor,
    batch_size: int,
    use_last: bool,
    device: torch.device,
) -> torch.Tensor:
    if mask.ndim == 1:
        if int(mask.shape[0]) != batch_size:
            raise ValueError(
                f"Padding mask batch mismatch: expected {batch_size}, got {int(mask.shape[0])}"
            )
        return mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2:
        if int(mask.shape[0]) != batch_size:
            raise ValueError(
                f"Padding mask batch mismatch: expected {batch_size}, got {int(mask.shape[0])}"
            )
        idx = -1 if use_last else 0
        return mask[:, idx].to(device=device, dtype=torch.bool)
    raise ValueError(f"Unsupported padding mask shape: {tuple(mask.shape)}")


def prepare_image_inputs(
    *,
    batch: Stage2Batch,
    device: torch.device,
    config: ImageTransformConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    streams = batch.image_streams
    if streams is None:
        raise ValueError("Expected image_streams in Stage2Batch.")
    mask_map = batch.image_padding_masks
    if mask_map is None:
        raise ValueError("Expected image_padding_masks in Stage2Batch.")

    configured_keys: list[str]
    if config.camera_keys is None:
        configured_keys = list(streams.keys())
    else:
        configured_keys = [str(k) for k in config.camera_keys]

    present = [k for k in configured_keys if k in streams]
    missing = [k for k in configured_keys if k not in streams]
    if not present and int(config.empty_cameras) <= 0:
        raise ValueError(
            f"No configured camera stream found. configured={configured_keys}, available={list(streams.keys())}"
        )

    images: list[torch.Tensor] = []
    img_masks: list[torch.Tensor] = []

    for key in present:
        stream = streams[key]
        use_last = True
        image = _select_image_step(stream, use_last=use_last)
        image = _to_bchw(image).to(device=device)
        image = gpu_preprocess_images(
            image,
            target_size=tuple(config.image_size),
            normalize=bool(config.normalize_to_neg_one_to_one),
        )

        mask = _extract_padding_mask(
            mask=mask_map[key],
            batch_size=int(image.shape[0]),
            use_last=use_last,
            device=device,
        )
        images.append(image)
        img_masks.append(mask)

    num_empty = min(len(missing), max(0, int(config.empty_cameras)))
    if num_empty > 0:
        if not images:
            raise ValueError(
                "empty_cameras>0 requires at least one present camera to infer tensor shape."
            )
        for _ in range(num_empty):
            fill = -1.0 if bool(config.normalize_to_neg_one_to_one) else 0.0
            images.append(torch.full_like(images[0], fill))
            img_masks.append(torch.zeros_like(img_masks[0]))

    return images, img_masks


def prepare_state_tensor(
    *,
    batch: Stage2Batch,
    device: torch.device,
    dtype: torch.dtype,
    max_state_dim: int,
) -> torch.Tensor:
    state = batch.state

    if state.ndim == 3:
        state = state[:, -1, :]
    if state.ndim != 2:
        raise ValueError(f"state must be rank 2 or 3, got shape {tuple(state.shape)}")

    state = state.to(device=device, dtype=dtype)
    return pad_vector(state, int(max_state_dim))


def to_action_chunk(
    *,
    actions: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if actions.ndim == 2:
        if int(chunk_size) != 1:
            raise ValueError(
                f"Expected action tensor [B,{int(chunk_size)},A] when chunk_size>1, got [B,A]={tuple(actions.shape)}"
            )
        return actions[:, None, :]
    if actions.ndim != 3:
        raise ValueError(
            f"Expected action tensor [B,A] or [B,T,A], got {tuple(actions.shape)}"
        )
    if int(actions.shape[1]) != int(chunk_size):
        raise ValueError(
            f"Expected action tensor [B,{int(chunk_size)},A], got {tuple(actions.shape)}"
        )
    return actions


def resolve_action_pad_field(
    *,
    batch: Mapping[str, Any],
    action_is_pad_key: str,
    actions_id_pad_key: str,
) -> torch.Tensor:
    has_action_is_pad = action_is_pad_key in batch
    has_actions_id_pad = actions_id_pad_key in batch
    if not has_action_is_pad and not has_actions_id_pad:
        raise KeyError(
            f"Missing action pad mask. Expected one of keys: {action_is_pad_key!r}, {actions_id_pad_key!r}"
        )

    if has_action_is_pad and has_actions_id_pad:
        action_is_pad = batch[action_is_pad_key]
        actions_id_pad = batch[actions_id_pad_key]
        action_is_pad_t = (
            action_is_pad
            if torch.is_tensor(action_is_pad)
            else torch.as_tensor(action_is_pad, dtype=torch.bool)
        )
        actions_id_pad_t = (
            actions_id_pad
            if torch.is_tensor(actions_id_pad)
            else torch.as_tensor(actions_id_pad, dtype=torch.bool)
        )
        action_is_pad_t = action_is_pad_t.to(dtype=torch.bool)
        actions_id_pad_t = actions_id_pad_t.to(dtype=torch.bool)
        if action_is_pad_t.shape != actions_id_pad_t.shape or not torch.equal(
            action_is_pad_t, actions_id_pad_t
        ):
            raise ValueError(
                f"Conflicting action pad masks in {action_is_pad_key!r} and {actions_id_pad_key!r}"
            )
        return action_is_pad_t

    mask = batch[action_is_pad_key] if has_action_is_pad else batch[actions_id_pad_key]
    return mask if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=torch.bool)


def resolve_action_pad_mask(
    *,
    action_is_pad: torch.Tensor,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if action_is_pad.ndim != 2:
        raise ValueError(
            f"Expected action_is_pad [B,T], got {tuple(action_is_pad.shape)}"
        )
    if int(action_is_pad.shape[0]) != int(batch_size):
        raise ValueError(
            f"action_is_pad batch mismatch: expected {batch_size}, got {int(action_is_pad.shape[0])}"
        )
    if int(action_is_pad.shape[1]) != int(chunk_size):
        raise ValueError(
            f"action_is_pad time mismatch: expected {chunk_size}, got {int(action_is_pad.shape[1])}"
        )
    return action_is_pad.to(device=device, dtype=torch.bool)


def _resolve_stats_entry(
    *,
    stats: Mapping[str, Mapping[str, Any]] | None,
    candidates: Sequence[str],
) -> Mapping[str, Any] | None:
    if stats is None:
        return None
    for key in candidates:
        if key in stats:
            return stats[key]
    return None


def normalize_vector_mean_std(
    *,
    value: torch.Tensor,
    stats: Mapping[str, Mapping[str, Any]] | None,
    key_candidates: Sequence[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    entry = _resolve_stats_entry(stats=stats, candidates=key_candidates)
    if entry is None:
        return value
    mean = entry.get("mean")
    std = entry.get("std")
    if mean is None or std is None:
        return value
    mean_t = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    std_t = torch.as_tensor(std, device=value.device, dtype=value.dtype)
    if mean_t.ndim == 1 and std_t.ndim == 1 and value.shape[-1] != mean_t.shape[0]:
        target_dim = int(value.shape[-1])
        src_dim = int(mean_t.shape[0])
        if src_dim > target_dim:
            pad_shape = (*value.shape[:-1], src_dim - target_dim)
            pad_value = mean_t[target_dim:].view(
                *([1] * (value.ndim - 1)), src_dim - target_dim
            )
            value = torch.cat([value, pad_value.expand(pad_shape)], dim=-1)
            target_dim = src_dim
        mean_pad = torch.zeros((target_dim,), device=value.device, dtype=value.dtype)
        std_pad = torch.ones((target_dim,), device=value.device, dtype=value.dtype)
        mean_pad[:src_dim] = mean_t
        std_pad[:src_dim] = std_t
        mean_t = mean_pad
        std_t = std_pad
    return (value - mean_t) / (std_t + float(eps))


def unnormalize_vector_mean_std(
    *,
    value: torch.Tensor,
    stats: Mapping[str, Mapping[str, Any]] | None,
    key_candidates: Sequence[str],
) -> torch.Tensor:
    entry = _resolve_stats_entry(stats=stats, candidates=key_candidates)
    if entry is None:
        return value
    mean = entry.get("mean")
    std = entry.get("std")
    if mean is None or std is None:
        return value
    mean_t = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    std_t = torch.as_tensor(std, device=value.device, dtype=value.dtype)
    if mean_t.ndim == 1 and std_t.ndim == 1 and value.shape[-1] != mean_t.shape[0]:
        target_dim = int(value.shape[-1])
        src_dim = int(mean_t.shape[0])
        if src_dim > target_dim:
            pad_value = torch.zeros(
                (*value.shape[:-1], src_dim - target_dim),
                device=value.device,
                dtype=value.dtype,
            )
            value = torch.cat([value, pad_value], dim=-1)
            target_dim = src_dim
        mean_pad = torch.zeros((target_dim,), device=value.device, dtype=value.dtype)
        std_pad = torch.ones((target_dim,), device=value.device, dtype=value.dtype)
        mean_pad[:src_dim] = mean_t
        std_pad[:src_dim] = std_t
        mean_t = mean_pad
        std_t = std_pad
    return value * std_t + mean_t
