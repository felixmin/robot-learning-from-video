from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from lam.models.forward_core import EncodedBatch


def compute_dino_loss(
    model: Any,
    *,
    encoded: EncodedBatch,
    action_tokens: torch.Tensor,
    attn_bias: torch.Tensor,
    h_dec: int,
    w_dec: int,
    step: int,
    metrics: Dict[str, Any],
) -> Optional[torch.Tensor]:
    if model.dino_decoder is None:
        return None

    dino_weight = model.dino_config.get_weight(step)
    metrics["dino_weight"] = dino_weight
    if dino_weight == 0.0:
        return None

    dino_context = encoded.enc_first_frame_tokens
    video_shape = tuple(dino_context.shape[:-1])
    dino_context_flat = rearrange(dino_context, "b t h w d -> (b t) (h w) d")
    dino_action_flat = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")
    pred_dino_tokens = model.dino_decoder(
        dino_context_flat,
        attn_bias=attn_bias,
        video_shape=video_shape,
        context=dino_action_flat,
    )
    pred_dino_tokens = rearrange(
        pred_dino_tokens,
        "(b t) (h w) d -> b t h w d",
        b=encoded.batch_size,
        h=h_dec,
        w=w_dec,
    )
    target_dino_tokens = encoded.enc_rest_frames_tokens.detach()
    dino_loss = F.mse_loss(pred_dino_tokens, target_dino_tokens)
    metrics["dino_loss"] = dino_loss.detach()
    return dino_weight * dino_loss


def compute_pixel_loss(
    model: Any,
    *,
    rest_frames: torch.Tensor,
    action_tokens: torch.Tensor,
    pixel_context: Optional[torch.Tensor],
    attn_bias: torch.Tensor,
    h_dec: int,
    w_dec: int,
    batch_size: int,
    metrics: Dict[str, Any],
) -> Optional[torch.Tensor]:
    if model.pixel_decoder is None:
        return None
    if pixel_context is None:
        raise RuntimeError("pixel_context is required when pixel decoder is enabled")

    video_shape = tuple(pixel_context.shape[:-1])
    pixel_context_flat = rearrange(pixel_context, "b t h w d -> (b t) (h w) d")
    pixel_action_flat = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")
    pred_pixel_tokens = model.pixel_decoder(
        pixel_context_flat,
        attn_bias=attn_bias,
        video_shape=video_shape,
        context=pixel_action_flat,
    )
    pred_pixel_tokens = rearrange(
        pred_pixel_tokens,
        "(b t) (h w) d -> b t h w d",
        b=batch_size,
        h=h_dec,
        w=w_dec,
    )
    pred_pixels = model.pixel_to_pixels(pred_pixel_tokens)
    pixel_loss = F.mse_loss(rest_frames, pred_pixels)
    metrics["pixel_loss"] = pixel_loss.detach()
    return pixel_loss


def compute_aux_outputs(
    model: Any,
    *,
    rest_frames: torch.Tensor,
    action_tokens: torch.Tensor,
    pixel_context: Optional[torch.Tensor],
    attn_bias: torch.Tensor,
    h_dec: int,
    w_dec: int,
    batch_size: int,
    return_recons_only: bool,
    metrics: Dict[str, Any],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if model.aux_decoder is None:
        if return_recons_only:
            return None, None
        return None, None
    if pixel_context is None:
        raise RuntimeError("pixel_context is required when aux decoder is enabled")

    aux_actions = action_tokens.detach()
    video_shape = tuple(pixel_context.shape[:-1])
    aux_context_flat = rearrange(pixel_context, "b t h w d -> (b t) (h w) d")
    aux_action_flat = rearrange(aux_actions, "b t h w d -> (b t) (h w) d")
    aux_tokens = model.aux_decoder(
        aux_context_flat,
        attn_bias=attn_bias,
        video_shape=video_shape,
        context=aux_action_flat,
    )
    aux_tokens = rearrange(
        aux_tokens,
        "(b t) (h w) d -> b t h w d",
        b=batch_size,
        h=h_dec,
        w=w_dec,
    )
    recon_video = model.aux_to_pixels(aux_tokens)

    if return_recons_only:
        recon_frames = rearrange(recon_video, "b c 1 h w -> b c h w")
        return None, recon_frames

    aux_loss = F.mse_loss(rest_frames, recon_video)
    metrics["aux_loss"] = aux_loss.detach()
    return aux_loss, None


def compute_flow_loss_term(
    model: Any,
    *,
    first_frame: torch.Tensor,
    rest_frames: torch.Tensor,
    action_tokens: torch.Tensor,
    pixel_context: Optional[torch.Tensor],
    attn_bias: torch.Tensor,
    step: int,
    metrics: Dict[str, Any],
) -> Optional[torch.Tensor]:
    if model.flow_decoder is None:
        return None

    from lam.models.flow import (
        compute_flow_loss,
        compute_flow_summary_loss,
        compute_weighted_mean_flow,
    )

    flow_weight = model.flow_config.get_weight(step)
    metrics["flow_weight"] = flow_weight
    metrics["flow_summary_weight"] = model.flow_config.summary_loss_weight
    if flow_weight == 0.0:
        return None
    if pixel_context is None:
        raise RuntimeError("pixel_context is required when flow decoder is enabled")

    gt_flow = model.flow_teacher.compute_flow(first_frame, rest_frames)
    pred_flow = model.flow_decoder(pixel_context, action_tokens, attn_bias)

    flow_loss = compute_flow_loss(pred_flow, gt_flow)
    loss_term = flow_weight * flow_loss
    metrics["flow_loss"] = flow_loss.detach()

    if model.flow_config.summary_loss_weight > 0.0:
        flow_summary_loss = compute_flow_summary_loss(
            pred_flow=pred_flow,
            gt_flow=gt_flow,
            static_eps=model.flow_config.summary_static_eps,
        )
        loss_term = loss_term + (
            flow_weight * model.flow_config.summary_loss_weight * flow_summary_loss
        )
        metrics["flow_summary_loss"] = flow_summary_loss.detach()

        pred_mean_dx, pred_mean_dy = compute_weighted_mean_flow(
            pred_flow,
            static_eps=model.flow_config.summary_static_eps,
        )
        gt_mean_dx, gt_mean_dy = compute_weighted_mean_flow(
            gt_flow,
            static_eps=model.flow_config.summary_static_eps,
        )
        metrics["flow_mean_dx_abs_err"] = (
            (pred_mean_dx - gt_mean_dx).abs().mean().detach()
        )
        metrics["flow_mean_dy_abs_err"] = (
            (pred_mean_dy - gt_mean_dy).abs().mean().detach()
        )

    return loss_term
