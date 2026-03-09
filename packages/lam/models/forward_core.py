import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


@dataclass
class EncodedBatch:
    batch_size: int
    device: torch.device
    first_frame: torch.Tensor
    rest_frames: torch.Tensor
    enc_first_frame_tokens: torch.Tensor
    enc_rest_frames_tokens: torch.Tensor
    tokens: torch.Tensor
    indices: torch.Tensor
    perplexity: torch.Tensor


@dataclass
class CodebookStats:
    current_threshold: float
    window_total: torch.Tensor
    codebook_replaced: float
    replaced_count: float
    used_count: float
    min_count: float


def normalize_video_input(model: Any, video: torch.Tensor) -> torch.Tensor:
    if video.ndim not in {4, 5}:
        raise ValueError(f"Expected 4D or 5D input, got {video.ndim}D")
    if video.ndim == 4:
        video = rearrange(video, "b c h w -> b c 1 h w")
    if tuple(video.shape[3:]) != model.image_size:
        raise ValueError(
            f"Expected image size {model.image_size}, got {tuple(video.shape[3:])}"
        )
    return video


def encode_and_quantize(model: Any, video: torch.Tensor) -> EncodedBatch:
    first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
    enc_first_frame_tokens, enc_rest_frames_tokens, first_tokens, last_tokens = (
        model._encode_frames(first_frame, rest_frames)
    )
    tokens, perplexity, _, indices = model.vq(
        first_tokens,
        last_tokens,
        codebook_training_only=False,
    )
    return EncodedBatch(
        batch_size=int(video.shape[0]),
        device=video.device,
        first_frame=first_frame,
        rest_frames=rest_frames,
        enc_first_frame_tokens=enc_first_frame_tokens,
        enc_rest_frames_tokens=enc_rest_frames_tokens,
        tokens=tokens,
        indices=indices,
        perplexity=perplexity,
    )


def prepare_action_tokens(model: Any, encoded: EncodedBatch) -> torch.Tensor:
    action_h, action_w = model.action_shape
    action_tokens = rearrange(
        encoded.tokens,
        "b (t h w) d -> b t h w d",
        h=action_h,
        w=action_w,
    )
    if model.latent_ablation == "permute_batch" and encoded.batch_size > 1:
        perm = torch.randperm(encoded.batch_size, device=action_tokens.device)
        action_tokens = action_tokens[perm]
    return action_tokens


def update_codebook_stats(model: Any, step: int) -> CodebookStats:
    step_i = int(step)
    current_vq_discarding_threshold = model._get_vq_discarding_threshold(step_i)
    window_total = model.vq.codebooks_used.sum().detach()
    unused_indices, used_indices, min_count_val = model.vq._get_replacement_indices(
        discarding_threshold=current_vq_discarding_threshold
    )

    codebook_replaced = 0.0
    replaced_count = float(unused_indices.shape[0])
    used_count = float(used_indices.shape[0])
    min_count = float(min_count_val)

    if step_i != 0 and model._should_replace_codebook(step_i):
        logger.debug(f"Replacing unused codebook entries at step {step_i}")
        codebook_replaced = 1.0
        unused_c, used_c, total_c, min_c = model.vq.replace_unused_codebooks(
            discarding_threshold=current_vq_discarding_threshold
        )
        replaced_count = float(unused_c)
        used_count = float(used_c)
        min_count = float(min_c)
        window_total = window_total.new_tensor(float(total_c))
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[Codebook] step={step_i} replaced={int(replaced_count)} "
                f"used={int(used_count)} total={int(total_c)} min_count={min_count:.4f} "
                f"threshold={current_vq_discarding_threshold:.6f}"
            )

    return CodebookStats(
        current_threshold=current_vq_discarding_threshold,
        window_total=window_total,
        codebook_replaced=codebook_replaced,
        replaced_count=replaced_count,
        used_count=used_count,
        min_count=min_count,
    )


def initialize_metrics(
    model: Any,
    *,
    perplexity: torch.Tensor,
    indices: torch.Tensor,
    step: int,
    codebook: CodebookStats,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "code_usage_perplexity_in_batch": perplexity.detach(),
        "replacement_applied_this_step": codebook.codebook_replaced,
        "entries_below_usage_threshold_count_in_window": codebook.replaced_count,
        "entries_at_or_above_usage_threshold_count_in_window": codebook.used_count,
        "usage_count_threshold_in_window": codebook.min_count,
        "discarding_threshold_fraction_of_expected_usage": codebook.current_threshold,
        "code_assignments_in_window": codebook.window_total,
    }
    if (
        model.metrics_num_unique_codes_every_n_steps > 0
        and int(step) % int(model.metrics_num_unique_codes_every_n_steps) == 0
    ):
        metrics["unique_codes_in_batch"] = int(indices.unique().size(0))
    return metrics


def build_decoder_shared_inputs(
    model: Any,
    *,
    first_frame: torch.Tensor,
    device: torch.device,
) -> tuple[int, int, torch.Tensor, Optional[torch.Tensor]]:
    h_dec, w_dec = model.patch_height_width
    attn_bias = model.spatial_rel_pos_bias(h_dec, w_dec, device=device)

    pixel_context = None
    if (
        model.pixel_decoder is not None
        or model.aux_decoder is not None
        or model.flow_decoder is not None
    ):
        pixel_context = model.decoder_context_projection(first_frame)
    return h_dec, w_dec, attn_bias, pixel_context
