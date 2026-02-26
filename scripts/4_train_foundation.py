#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 4: Train Foundation VLA Model

Train the foundation VLA model (Stage 2): image + language -> latent action tokens.

Usage:
    # Debug (local / single GPU):
    python scripts/4_train_foundation.py experiment=vla_cosmos2_tokens_debug model.laq.checkpoint=/path/to/laq.ckpt
"""

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.callbacks import DatasetUsageLoggerCallback, ProgressLoggerCallback
from common.cache_env import configure_cache_env, hf_download_help_message, resolve_cache_dir
from common.data_factory import create_datamodule
from common.logging import set_seed
from common.unified_logging import resolve_runs_dir, setup_unified_logging, setup_wandb_with_unified_paths
from foundation.action_tokens import ActionTokenConfig
from foundation.callbacks import (
    ThroughputLoggingCallback,
    ThroughputLoggingConfig,
    VLATrainSampleVizConfig,
    VLATrainSampleVisualizationCallback,
    VLASampleVizConfig,
    VLASampleVisualizationCallback,
)
from foundation.backends.interfaces import BackendMode
from foundation.image_adapters import oxe_first_frames_to_pil
# Legacy backends kept for migration/reference.
from foundation.legacy.backends.qwen3vl_chat_backend import (
    Qwen3VLChatActionTokenBackend,
    Qwen3VLChatBackendConfig,
)
from foundation.legacy.backends.smol_latent_head_backend import (
    SmolFlowActionBackend,
    SmolFlowActionBackendConfig,
    SmolLatentHeadBackend,
    SmolLatentHeadBackendConfig,
)
from foundation.backends.smolvla_shared.artifact import (
    SMOLVLA_SHARED_ARTIFACT_FILENAME,
    SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION,
    SmolVLASharedArtifactManifest,
    save_smolvla_shared_artifact,
)
from foundation.backends.smolvla_shared.config import SmolVLASharedBackendConfig
from foundation.backends.smolvla_shared_backend import SmolVLASharedBackend
from foundation.online_laq import LAQTaskCodeProvider
from foundation.vla_inputs import ChatConfig
from foundation.vla_backend_module import VLATokenBackendLightningModule, VLAOptimizerConfig
from laq.checkpoints import load_laq_encoder_vq_inference_from_checkpoint


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Force new W&B run per Hydra job (fix for sweeps)
    if wandb.run:
        wandb.finish()

    # Setup unified logging
    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None
    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=cfg.logging.root_dir,
            logging_runs_dir=cfg.logging.runs_dir,
            workspace_root=workspace_root,
            experiment_name=OmegaConf.select(cfg, "experiment.name"),
        )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.job_id,
        log_level=cfg.logging.level,
        logger_name="foundation.training",
    )

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    logger.info("=" * 80)
    logger.info("LAPA Stage 2: Foundation VLA Training (Action Tokens)")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # This script currently implements the Lightning training stack (Stage 2, tokens).
    # Fail fast if an incompatible training config (e.g., legacy FSDP schema) is composed.
    required_training_keys = [
        "max_steps",
        "max_epochs",
        "accumulate_grad_batches",
        "precision",
        "gradient_clip_val",
        "log_every_n_steps",
        "optimizer",
        "validation",
        "checkpoint",
        "train_visualization",
        "throughput",
        "dataset_usage_logger",
        "progress_logger",
        "resume_from_checkpoint",
    ]
    missing = [k for k in required_training_keys if k not in cfg.training]
    if missing:
        raise RuntimeError(
            "Unsupported training config for `scripts/4_train_foundation.py`. "
            "Expected Lightning-style `training` schema. "
            f"Missing keys: {missing}"
        )

    set_seed(int(cfg.seed))

    # Setup WandB logger (use unified logging paths). If disabled, avoid Lightning's default logger
    # to prevent creating extra `lightning_logs/` directories.
    # Calculate unique run name for sweeps
    run_name = cfg.experiment.name
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == "MULTIRUN":
            run_name = f"{run_name}_{hydra_cfg.job.num}"
    except (ValueError, Exception):
        pass

    wandb_logger = setup_wandb_with_unified_paths(
        logger=logger,
        output_dir=output_dir,
        project=cfg.logging.project,
        name=run_name,
        group=cfg.experiment.name,
        tags=cfg.logging.tags,
        use_wandb=bool(cfg.logging.use_wandb),
        settings=wandb.Settings(start_method="thread", reinit=True),
    )

    # Data: frame pairs + language.
    if cfg.data.backend != "oxe_local_indexed":
        raise ValueError(
            "Stage 2 expects data.backend='oxe_local_indexed', "
            f"got {cfg.data.backend!r}"
        )

    datamodule = create_datamodule(cfg.data)
    datamodule.setup()

    # LAQ: frozen label generator (encoder+VQ only; decoders pruned to reclaim VRAM)
    laq_ckpt = cfg.model.laq.checkpoint
    if not laq_ckpt:
        raise ValueError(
            "Set `model.laq.checkpoint=/path/to/laq.ckpt` for online LAQ labeling."
        )
    try:
        encoder_vq = load_laq_encoder_vq_inference_from_checkpoint(laq_ckpt)
    except Exception as exc:
        help_msg = hf_download_help_message(exc=exc)
        if help_msg:
            logger.error(help_msg)
        raise RuntimeError(
            f"Failed to load LAQ checkpoint '{laq_ckpt}'. "
            "Provide a checkpoint trained with the current LAQ codebase."
        ) from exc
    laq_provider = LAQTaskCodeProvider(encoder_vq)

    backend_type = OmegaConf.select(cfg, "model.backend")
    if not backend_type:
        raise ValueError("Set `model.backend` (e.g., 'smolvla_shared').")
    backend_type = str(backend_type)
    backend_mode_raw = OmegaConf.select(cfg, "model.training_mode")
    if not backend_mode_raw:
        raise ValueError("Set `model.training_mode` to one of: codes, latent_flow, actions, multitask.")
    try:
        backend_mode = BackendMode(str(backend_mode_raw))
    except ValueError as exc:
        raise ValueError(
            f"Unknown model.training_mode={backend_mode_raw!r}. Use one of: codes, latent_flow, actions, multitask."
        ) from exc

    model_name = cfg.model.vla.model_name
    torch_dtype = str(cfg.model.vla.torch_dtype).lower()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if torch_dtype not in dtype_map:
        raise ValueError(
            f"Unknown model.vla.torch_dtype={torch_dtype!r}. "
            f"Supported: {sorted(dtype_map.keys())}"
        )
    dtype = dtype_map[torch_dtype]

    action_cfg = ActionTokenConfig(
        **OmegaConf.to_container(cfg.model.action_tokens, resolve=True)
    )

    action_token_ids = None
    if backend_type == "qwen3vl_chat_tokens":
        if backend_mode is not BackendMode.CODES:
            raise ValueError("qwen3vl_chat_tokens backend currently supports only model.training_mode=codes")
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        vla_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=cfg.model.vla.attn_implementation,
        )
        vla_model.train()
        processor = Qwen3VLProcessor.from_pretrained(model_name)

        backend = Qwen3VLChatActionTokenBackend(
            config=Qwen3VLChatBackendConfig(
                model_name=str(model_name),
                torch_dtype=dtype,
                attn_implementation=cfg.model.vla.attn_implementation,
                chat=ChatConfig(system_prompt=cfg.model.chat.system_prompt),
                action_tokens=action_cfg,
            ),
            vla_model=vla_model,
            processor=processor,
            frames_to_images=oxe_first_frames_to_pil,
        )
        backend.setup(device=torch.device("cpu"))
        action_token_ids = backend.action_token_ids

        code_ids = list(action_token_ids.action_code_ids)
        if len(set(code_ids)) != len(code_ids):
            raise RuntimeError(
                "Action code token ids are not unique. "
                "This typically means the action tokens were not properly added to the tokenizer. "
                f"code_ids={code_ids}"
            )
        between_ids = list(action_token_ids.between_token_ids)
        if not between_ids:
            raise RuntimeError(
                "between_token_ids is empty. Constrained decoding must allow separator tokens "
                "to match the supervised target format."
            )
        if any(int(t) in set(code_ids) for t in between_ids):
            raise RuntimeError(
                "One or more between_token_ids overlaps with an action code token id; "
                "this will confuse constrained decoding. "
                f"between_token_ids={between_ids} code_ids={code_ids}"
            )
        if action_token_ids.action_start_id in code_ids or action_token_ids.action_end_id in code_ids:
            raise RuntimeError(
                "Action wrapper token id overlaps with an action code token id. "
                f"start_id={action_token_ids.action_start_id} end_id={action_token_ids.action_end_id} code_ids={code_ids}"
            )
        if action_token_ids.action_start_id in between_ids or action_token_ids.action_end_id in between_ids:
            raise RuntimeError(
                "between_token_ids overlaps with an action wrapper token id; "
                f"start_id={action_token_ids.action_start_id} end_id={action_token_ids.action_end_id} between_token_ids={between_ids}"
            )
        unk_id = getattr(processor.tokenizer, "unk_token_id", None)
        if unk_id is not None and unk_id in code_ids:
            raise RuntimeError(
                "One or more action code tokens mapped to unk_token_id; tokenization will be broken. "
                f"unk_token_id={unk_id} code_ids={code_ids}"
            )
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None and int(pad_id) in code_ids:
            raise RuntimeError(
                "pad_token_id overlaps with an action code token id; this will break decoding/parsing. "
                f"pad_token_id={int(pad_id)} code_ids={code_ids}"
            )

    elif backend_type == "smol_latent_head":
        if backend_mode is not BackendMode.CODES:
            raise ValueError("smol_latent_head backend currently supports only model.training_mode=codes")
        trust_remote_code = bool(OmegaConf.select(cfg, "model.vla.trust_remote_code") or False)
        use_gpu_preprocessing = bool(OmegaConf.select(cfg, "model.vla.use_gpu_preprocessing") or False)
        image_size_cfg = OmegaConf.select(cfg, "model.vla.image_size")
        image_size = tuple(image_size_cfg) if image_size_cfg else (384, 384)
        backend = SmolLatentHeadBackend(
            config=SmolLatentHeadBackendConfig(
                model_name=str(model_name),
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                chat=ChatConfig(system_prompt=cfg.model.chat.system_prompt),
                action_tokens=action_cfg,
                use_gpu_preprocessing=use_gpu_preprocessing,
                image_size=image_size,
            ),
            frames_to_images=oxe_first_frames_to_pil,
        )
        try:
            backend.setup(device=torch.device("cpu"))
        except Exception as exc:
            help_msg = hf_download_help_message(exc=exc)
            if help_msg:
                logger.error(help_msg)
            raise
    elif backend_type == "smol_flow_action":
        if backend_mode not in (BackendMode.MULTITASK, BackendMode.ACTIONS):
            raise ValueError("smol_flow_action backend supports model.training_mode in {actions, multitask}")

        trust_remote_code = bool(OmegaConf.select(cfg, "model.vla.trust_remote_code") or False)
        use_gpu_preprocessing = bool(OmegaConf.select(cfg, "model.vla.use_gpu_preprocessing") or False)
        image_size_cfg = OmegaConf.select(cfg, "model.vla.image_size")
        image_size = tuple(image_size_cfg) if image_size_cfg else (384, 384)
        flow_cfg = OmegaConf.select(cfg, "model.flow")
        if flow_cfg is None:
            raise ValueError("Missing model.flow config for smol_flow_action backend")

        latent_vector_dim = int(laq_provider.code_seq_len * laq_provider.codebook_dim)
        backend = SmolFlowActionBackend(
            config=SmolFlowActionBackendConfig(
                model_name=str(model_name),
                latent_vector_dim=latent_vector_dim,
                action_dim=int(flow_cfg.action_dim),
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                chat=ChatConfig(system_prompt=cfg.model.chat.system_prompt),
                action_tokens=action_cfg,
                use_gpu_preprocessing=use_gpu_preprocessing,
                image_size=image_size,
                flow_hidden_dim=int(flow_cfg.flow_hidden_dim),
                flow_steps=int(flow_cfg.flow_steps),
                latent_loss_weight=float(flow_cfg.latent_loss_weight),
                action_loss_weight=float(flow_cfg.action_loss_weight),
            ),
            frames_to_images=oxe_first_frames_to_pil,
        )
        try:
            backend.setup(device=torch.device("cpu"))
        except Exception as exc:
            help_msg = hf_download_help_message(exc=exc)
            if help_msg:
                logger.error(help_msg)
            raise
    elif backend_type == "smolvla_shared":
        if backend_mode not in (BackendMode.LATENT_FLOW, BackendMode.ACTIONS, BackendMode.MULTITASK):
            raise ValueError(
                "smolvla_shared backend supports model.training_mode in {latent_flow, actions, multitask}"
            )

        trust_remote_code = bool(OmegaConf.select(cfg, "model.vla.trust_remote_code") or False)
        use_gpu_preprocessing = bool(OmegaConf.select(cfg, "model.vla.use_gpu_preprocessing") or False)
        freeze_vlm_cfg = OmegaConf.select(cfg, "model.vla.freeze_vlm")
        freeze_vlm = True if freeze_vlm_cfg is None else bool(freeze_vlm_cfg)
        image_size_cfg = OmegaConf.select(cfg, "model.vla.image_size")
        image_size = tuple(image_size_cfg) if image_size_cfg else (384, 384)
        flow_cfg = OmegaConf.select(cfg, "model.flow")
        if flow_cfg is None:
            raise ValueError("Missing model.flow config for smolvla_shared backend")

        latent_vector_dim = int(laq_provider.code_seq_len * laq_provider.codebook_dim)
        action_dim_cfg = OmegaConf.select(flow_cfg, "action_dim")
        action_dim = int(action_dim_cfg) if action_dim_cfg is not None else None

        backend_kwargs = dict(
            model_name=str(model_name),
            latent_vector_dim=latent_vector_dim,
            action_dim=action_dim,
            freeze_vlm=freeze_vlm,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            chat=ChatConfig(system_prompt=cfg.model.chat.system_prompt),
            action_tokens=action_cfg,
            use_gpu_preprocessing=use_gpu_preprocessing,
            image_size=image_size,
            flow_hidden_dim=int(flow_cfg.flow_hidden_dim),
            flow_steps=int(flow_cfg.flow_steps),
            latent_loss_weight=float(flow_cfg.latent_loss_weight),
            action_loss_weight=float(flow_cfg.action_loss_weight),
            min_period=float(OmegaConf.select(flow_cfg, "min_period") or 4e-3),
            max_period=float(OmegaConf.select(flow_cfg, "max_period") or 4.0),
            time_beta_alpha=float(OmegaConf.select(flow_cfg, "time_beta_alpha") or 1.5),
            time_beta_beta=float(OmegaConf.select(flow_cfg, "time_beta_beta") or 1.0),
        )

        optional_vla_overrides: dict[str, object] = {
            "freeze_vision_encoder": OmegaConf.select(cfg, "model.vla.freeze_vision_encoder"),
            "load_vlm_weights": OmegaConf.select(cfg, "model.vla.load_vlm_weights"),
            "attention_mode": OmegaConf.select(cfg, "model.vla.attention_mode"),
            "num_expert_layers": OmegaConf.select(cfg, "model.vla.num_expert_layers"),
            "num_vlm_layers": OmegaConf.select(cfg, "model.vla.num_vlm_layers"),
            "self_attn_every_n_layers": OmegaConf.select(cfg, "model.vla.self_attn_every_n_layers"),
            "expert_width_multiplier": OmegaConf.select(cfg, "model.vla.expert_width_multiplier"),
            "add_image_special_tokens": OmegaConf.select(cfg, "model.vla.add_image_special_tokens"),
            "tokenizer_max_length": OmegaConf.select(cfg, "model.vla.tokenizer_max_length"),
            "pad_language_to": OmegaConf.select(cfg, "model.vla.pad_language_to"),
            "max_state_dim": OmegaConf.select(cfg, "model.vla.max_state_dim"),
            "prefix_length": OmegaConf.select(cfg, "model.vla.prefix_length"),
        }
        for key, value in optional_vla_overrides.items():
            if value is not None:
                backend_kwargs[key] = value

        backend = SmolVLASharedBackend(
            config=SmolVLASharedBackendConfig(**backend_kwargs),
            frames_to_images=oxe_first_frames_to_pil,
        )
        try:
            backend.setup(device=torch.device("cpu"))
        except Exception as exc:
            help_msg = hf_download_help_message(exc=exc)
            if help_msg:
                logger.error(help_msg)
            raise
    else:
        raise ValueError(f"Unknown model.backend={backend_type!r}")

    module = VLATokenBackendLightningModule(
        backend=backend,
        code_provider=laq_provider,
        backend_mode=backend_mode,
        optimizer=VLAOptimizerConfig(
            lr=float(cfg.training.optimizer.lr),
            weight_decay=float(cfg.training.optimizer.weight_decay),
        ),
        action_token_ids=action_token_ids,
        train_teacher_forced_metrics_every_n_steps=(
            int(cfg.training.train_teacher_forced_metrics_every_n_steps)
            if cfg.training.train_teacher_forced_metrics_every_n_steps is not None
            else None
        ),
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_cfg = cfg.training.checkpoint
    every_n_train_steps = checkpoint_cfg.every_n_train_steps
    save_weights_only = bool(checkpoint_cfg.save_weights_only)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_top_k=int(checkpoint_cfg.save_top_k),
            save_last=bool(checkpoint_cfg.save_last),
            save_weights_only=save_weights_only,
            every_n_train_steps=int(every_n_train_steps) if every_n_train_steps is not None else None,
            filename="vla-step{step:06d}",
            verbose=True,
        ),
    ]
    viz_cfg = cfg.training.validation.visualization
    if viz_cfg and bool(viz_cfg.enabled):
        callbacks.append(
            VLASampleVisualizationCallback(
                VLASampleVizConfig(
                    enabled=True,
                    num_samples=int(viz_cfg.num_samples),
                    every_n_val=int(viz_cfg.every_n_val),
                    include_freeform_pred=bool(viz_cfg.include_freeform_pred),
                    freeform_max_new_tokens=int(viz_cfg.freeform_max_new_tokens),
                )
            )
        )
    train_viz_cfg = cfg.training.train_visualization
    if train_viz_cfg and bool(train_viz_cfg.enabled):
        callbacks.append(
            VLATrainSampleVisualizationCallback(
                VLATrainSampleVizConfig(
                    enabled=True,
                    num_samples=int(train_viz_cfg.num_samples),
                    every_n_steps=int(train_viz_cfg.every_n_steps),
                    include_freeform_pred=bool(train_viz_cfg.include_freeform_pred),
                    freeform_max_new_tokens=int(train_viz_cfg.freeform_max_new_tokens),
                )
            )
        )
    perf_cfg = cfg.training.throughput
    if perf_cfg and bool(perf_cfg.enabled):
        callbacks.append(
            ThroughputLoggingCallback(
                ThroughputLoggingConfig(
                    enabled=True,
                    log_every_n_steps=int(perf_cfg.log_every_n_steps),
                )
            )
        )

    # Progress logging (useful on clusters where tqdm doesn't render nicely in logs).
    # Default: enable on Slurm, disabled for local runs unless explicitly configured.
    progress_cfg = cfg.training.progress_logger
    enable_progress = bool(progress_cfg.enabled)
    if enable_progress:
        log_every = int(progress_cfg.log_every_n_steps)
        callbacks.append(ProgressLoggerCallback(log_every_n_steps=log_every))

    # Dataset usage logging (prints how much of each dataset was *actually consumed*).
    # Recommended to align with step-based validation cadence by logging on validation end.
    usage_cfg = cfg.training.dataset_usage_logger
    if usage_cfg and bool(usage_cfg.enabled):
        log_batch_mix_every = OmegaConf.select(
            usage_cfg, "log_batch_composition_every_n_steps", default=None
        )
        callbacks.append(
            DatasetUsageLoggerCallback(
                enabled=True,
                log_on_validation_end=bool(usage_cfg.log_on_validation_end),
                log_every_n_steps=usage_cfg.log_every_n_steps,
                log_batch_composition_every_n_steps=log_batch_mix_every,
                key=str(usage_cfg.key),
                top_k=int(usage_cfg.top_k),
            )
        )
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    else:
        logger.info("WandB disabled; skipping LearningRateMonitor (no logger).")

    # Validation defaults in Lightning run at the end of the (potentially huge) epoch.
    # For short max_steps runs with large IterableDatasets, validation may never run unless
    # we validate every N steps (like Stage 1) and/or limit validation batches.
    trainer_extra_kwargs: dict[str, object] = {}
    val_check_interval = cfg.training.validation.check_interval
    if val_check_interval is not None:
        trainer_extra_kwargs["val_check_interval"] = val_check_interval
    limit_val_batches = cfg.training.validation.limit_batches
    if limit_val_batches is not None:
        trainer_extra_kwargs["limit_val_batches"] = limit_val_batches
    num_sanity_val_steps = cfg.training.validation.num_sanity_val_steps
    if num_sanity_val_steps is not None:
        trainer_extra_kwargs["num_sanity_val_steps"] = int(num_sanity_val_steps)
    overfit_batches = cfg.training.overfit_batches
    if overfit_batches is not None:
        trainer_extra_kwargs["overfit_batches"] = overfit_batches
    limit_train_batches = cfg.training.limit_train_batches
    if limit_train_batches is not None:
        trainer_extra_kwargs["limit_train_batches"] = limit_train_batches

    # Optional profiler (matches Stage 1 conventions).
    profiler = None
    profiler_cfg = cfg.training.profiler
    if profiler_cfg and bool(profiler_cfg.enabled):
        profiler_type = str(profiler_cfg.type)
        dirpath = str(profiler_cfg.dirpath)
        dirpath_path = Path(dirpath)
        if not dirpath_path.is_absolute():
            # Match Stage 1 behavior: resolve relative profiler paths inside the run directory.
            dirpath_path = output_dir / dirpath_path
        dirpath = str(dirpath_path)
        filename = str(profiler_cfg.filename)
        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler

            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler

            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
        else:
            raise ValueError(f"Unknown profiler type: {profiler_type}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.training.validation.check_val_every_n_epoch,
        logger=wandb_logger if wandb_logger is not None else False,
        default_root_dir=str(output_dir),
        profiler=profiler,
        **trainer_extra_kwargs,
    )

    ckpt_path = cfg.training.resume_from_checkpoint
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if backend_type == "smolvla_shared" and trainer.is_global_zero:
        backend_cfg = module.backend.cfg
        artifact_path = output_dir / "artifacts" / SMOLVLA_SHARED_ARTIFACT_FILENAME
        manifest = SmolVLASharedArtifactManifest(
            schema_version=SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION,
            model_name=str(backend_cfg.model_name),
            torch_dtype=torch_dtype,
            image_size=(int(backend_cfg.image_size[0]), int(backend_cfg.image_size[1])),
            action_dim=(None if backend_cfg.action_dim is None else int(backend_cfg.action_dim)),
            latent_vector_dim=int(backend_cfg.latent_vector_dim),
            flow_hidden_dim=int(backend_cfg.flow_hidden_dim),
            flow_steps=int(backend_cfg.flow_steps),
            min_period=float(backend_cfg.min_period),
            max_period=float(backend_cfg.max_period),
            time_beta_alpha=float(backend_cfg.time_beta_alpha),
            time_beta_beta=float(backend_cfg.time_beta_beta),
            source_backend="smolvla_shared",
            source_training_mode=backend_mode.value,
            source_run_dir=str(output_dir),
            source_global_step=int(trainer.global_step),
        )
        save_smolvla_shared_artifact(
            path=artifact_path,
            manifest=manifest,
            core_state_dict=module.backend.core.state_dict(),
        )
        logger.info("Exported smolvla_shared stage2 artifact: %s", artifact_path)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
