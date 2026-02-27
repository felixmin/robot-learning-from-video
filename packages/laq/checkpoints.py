"""Shared LAQ checkpoint loading utilities."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Union

import torch
from omegaconf import OmegaConf

from laq.inference import LAQEncoderVQInference
from laq.models.flow import FlowConfig
from laq.models.latent_action_quantization import DinoConfig, LatentActionQuantization

if TYPE_CHECKING:
    from laq.task import LAQTask


def _load_checkpoint_dict(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")
    return checkpoint


def _extract_configs_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[Any, Any]:
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise KeyError("Checkpoint missing 'hyper_parameters' dict")

    model_config = hparams.get("model_config")
    training_config = hparams.get("training_config")
    if isinstance(model_config, dict):
        model_config = OmegaConf.create(model_config)
    if isinstance(training_config, dict):
        training_config = OmegaConf.create(training_config)
    if model_config is None or training_config is None:
        raise KeyError("Checkpoint hyper_parameters missing model_config or training_config")
    return model_config, training_config


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint missing 'state_dict'")
    return state_dict


def _build_laq_model_from_configs(*, model_config: Any, training_config: Any) -> LatentActionQuantization:
    flow_cfg = model_config.flow
    flow_enabled = bool(flow_cfg.enabled)
    flow_config = None
    if flow_enabled:
        flow_config = FlowConfig(
            model=flow_cfg.model,
            loss_weight=flow_cfg.loss_weight,
            decoder_depth=flow_cfg.decoder_depth,
            warmup_steps=flow_cfg.warmup_steps,
            teacher_num_flow_updates=flow_cfg.teacher_num_flow_updates,
            teacher_chunk_size=flow_cfg.teacher_chunk_size,
            summary_loss_weight=flow_cfg.summary_loss_weight,
            summary_static_eps=flow_cfg.summary_static_eps,
        )

    dino_cfg = model_config.dino
    dino_enabled = bool(dino_cfg.enabled)
    dino_config = None
    if dino_enabled:
        dino_config = DinoConfig(
            loss_weight=float(dino_cfg.loss_weight),
            warmup_steps=int(dino_cfg.warmup_steps),
        )

    codebook_replace_schedule = [
        tuple(entry) for entry in model_config.codebook_replace_schedule
    ]
    vq_discarding_threshold_schedule = None
    if (
        "vq_discarding_threshold_schedule" in model_config
        and model_config.vq_discarding_threshold_schedule is not None
    ):
        vq_discarding_threshold_schedule = [
            tuple(entry) for entry in model_config.vq_discarding_threshold_schedule
        ]

    metrics_cfg = training_config.metrics

    return LatentActionQuantization(
        dim=model_config.dim,
        quant_dim=model_config.quant_dim,
        codebook_size=model_config.codebook_size,
        image_size=model_config.image_size,
        patch_size=model_config.patch_size,
        spatial_depth=model_config.spatial_depth,
        temporal_depth=model_config.temporal_depth,
        dim_head=model_config.dim_head,
        heads=model_config.heads,
        code_seq_len=model_config.code_seq_len,
        vq_discarding_threshold=model_config.vq_discarding_threshold,
        vq_discarding_threshold_schedule=vq_discarding_threshold_schedule,
        channels=model_config.channels,
        attn_dropout=model_config.attn_dropout,
        ff_dropout=model_config.ff_dropout,
        latent_ablation=model_config.latent_ablation,
        use_dinov3_encoder=model_config.use_dinov3_encoder,
        dinov3_model_name=model_config.dinov3_model_name,
        dinov3_pool_to_grid=model_config.dinov3_pool_to_grid,
        metrics_num_unique_codes_every_n_steps=int(metrics_cfg.num_unique_codes_every_n_steps),
        dino_config=dino_config,
        use_dino_decoder=dino_enabled,
        use_pixel_decoder=model_config.use_pixel_decoder,
        use_aux_decoder=model_config.use_aux_decoder,
        flow_config=flow_config,
        codebook_replace_schedule=codebook_replace_schedule,
    )


def load_laq_task_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> "LAQTask":
    """
    Load a full LAQTask from a Lightning checkpoint.

    Tries LAQTask.load_from_checkpoint first; falls back to manual
    hparam extraction + state_dict load for cross-version compatibility.
    """
    from laq.task import LAQTask

    ckpt_path = str(checkpoint_path)

    try:
        load_kwargs: dict = {"map_location": map_location, "weights_only": False}
        if "strict" in inspect.signature(LAQTask.load_from_checkpoint).parameters:
            load_kwargs["strict"] = strict
        return LAQTask.load_from_checkpoint(ckpt_path, **load_kwargs)
    except TypeError:
        pass
    except RuntimeError as exc:
        if "weights_only" not in str(exc).lower():
            raise

    checkpoint = _load_checkpoint_dict(ckpt_path, map_location=map_location)
    model_config, training_config = _extract_configs_from_checkpoint(checkpoint)
    state_dict = _extract_model_state_dict(checkpoint)

    task = LAQTask(model_config=model_config, training_config=training_config)
    task.load_state_dict(state_dict, strict=strict)
    return task


def load_laq_model_weights_only(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    strip_model_prefix: bool = True,
    drop_optimizer_keys: bool = True,
) -> tuple[list[str], list[str], int]:
    """
    Load model weights from a checkpoint, ignoring optimizer/scheduler state.

    Returns (missing_keys, unexpected_keys, loaded_tensor_count).
    """
    load_kwargs: dict = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state_dict)}")

    if drop_optimizer_keys:
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith(("optimizer", "lr_scheduler"))
        }
    if strip_model_prefix:
        state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected), len(state_dict)


def load_laq_encoder_vq_inference_from_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    prune_decoders: bool = True,
) -> LAQEncoderVQInference:
    """
    Load a LAQEncoderVQInference from a checkpoint.

    Loads the full task then prunes decoder/teacher modules to reclaim VRAM.
    Caller is responsible for setting eval mode and moving to the target device.
    """
    checkpoint = _load_checkpoint_dict(checkpoint_path, map_location=map_location)
    model_config, training_config = _extract_configs_from_checkpoint(checkpoint)
    model = _build_laq_model_from_configs(model_config=model_config, training_config=training_config)
    state_dict = _extract_model_state_dict(checkpoint)
    model_state = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict LAQ model load failed for {checkpoint_path}. missing={list(missing)} unexpected={list(unexpected)}"
        )
    return LAQEncoderVQInference(model, prune_decoders=prune_decoders)
