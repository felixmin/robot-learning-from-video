#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 4: Train Stage2 Policy Model

Train the Stage 2 policy model: image + language -> latent action tokens.

Usage:
    # Debug (local / single GPU):
    python scripts/4_train_stage2_policy.py experiment=stage2_local model.stage1.checkpoint=/path/to/stage1_lam.ckpt
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

from common.callbacks import (
    DataSampleVisualizationCallback,
    DataSampleVisualizationConfig,
    DatasetUsageLoggerCallback,
    ProgressLoggerCallback,
)
from common.data_factory import create_datamodule
from common.logging import set_seed
from common.run_context import setup_run_context
from common.unified_logging import setup_wandb_with_unified_paths
from stage2.action_tokens import ActionTokenConfig
from stage2.callbacks import (
    ThroughputLoggingCallback,
    ThroughputLoggingConfig,
    PolicyTrainSampleVizConfig,
    PolicyTrainSampleVisualizationCallback,
)
from stage2.validation_pipeline_callback import Stage2ValidationPipelineCallback
from stage2.backends.smolvla_shared.artifact import (
    SMOLVLA_SHARED_ARTIFACT_FILENAME,
    SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION,
    SmolVLASharedArtifactManifest,
    save_smolvla_shared_artifact,
)
from stage2.backends.smolvla_shared.config import SmolVLASharedBackendConfig
from stage2.backends.smolvla_shared_backend import SmolVLASharedBackend
from stage2.online_lam import LAMTaskCodeProvider
from stage2.policy_inputs import ChatConfig
from stage2.policy_module import PolicyLightningModule, PolicyOptimizerConfig
from stage2.backends.interfaces import BackendMode
from lam import load_lam_encoder_vq_inference_from_checkpoint


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Force new W&B run per Hydra job (fix for sweeps)
    if wandb.run:
        wandb.finish()

    logger, output_dir = setup_run_context(
        cfg=cfg,
        workspace_root=workspace_root,
        logger_name="stage2.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 2: Policy Training (Action Tokens)")
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
            "Unsupported training config for `scripts/4_train_stage2_policy.py`. "
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
    datamodule = create_datamodule(cfg.data)
    datamodule.setup()

    backend_type = str(cfg.model.backend)
    backend_mode = BackendMode(str(cfg.model.training_mode))
    stage1_provider = None
    if backend_mode is not BackendMode.ACTIONS:
        stage1_ckpt = cfg.model.stage1.checkpoint
        if not stage1_ckpt:
            raise ValueError(
                "Set `model.stage1.checkpoint=/path/to/stage1_lam.ckpt` for online Stage-1 labeling."
            )
        encoder_vq = load_lam_encoder_vq_inference_from_checkpoint(stage1_ckpt)
        stage1_provider = LAMTaskCodeProvider(encoder_vq)

    model_name = cfg.model.policy.model_name

    torch_dtype = str(cfg.model.policy.torch_dtype).lower()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[torch_dtype]

    action_cfg = ActionTokenConfig(
        **OmegaConf.to_container(cfg.model.action_tokens, resolve=True)
    )

    action_token_ids = None
    if backend_type == "smolvla_shared":
        if backend_mode not in (BackendMode.LATENT_FLOW, BackendMode.ACTIONS, BackendMode.MULTITASK):
            raise ValueError(
                "smolvla_shared backend supports model.training_mode in {latent_flow, actions, multitask}"
            )

        trust_remote_code = bool(cfg.model.policy.trust_remote_code)
        use_gpu_preprocessing = bool(cfg.model.policy.use_gpu_preprocessing)
        freeze_vlm = bool(cfg.model.policy.freeze_vlm)
        image_size = tuple(int(x) for x in cfg.model.policy.image_size)
        flow_cfg = cfg.model.flow

        if stage1_provider is None:
            latent_vector_dim = int(cfg.stage2_policy_core.latent.latent_vector_dim)
        else:
            latent_vector_dim = int(stage1_provider.code_seq_len * stage1_provider.codebook_dim)
        action_dim = int(flow_cfg.action_dim)
        action_chunk_size = int(flow_cfg.action_chunk_size)

        backend_kwargs = dict(
            model_name=str(model_name),
            latent_vector_dim=latent_vector_dim,
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            freeze_vlm=freeze_vlm,
            freeze_vision_encoder=bool(cfg.model.policy.freeze_vision_encoder),
            load_vlm_weights=bool(cfg.model.policy.load_vlm_weights),
            attention_mode=str(cfg.model.policy.attention_mode),
            num_expert_layers=int(cfg.model.policy.num_expert_layers),
            num_vlm_layers=int(cfg.model.policy.num_vlm_layers),
            self_attn_every_n_layers=int(cfg.model.policy.self_attn_every_n_layers),
            expert_width_multiplier=float(cfg.model.policy.expert_width_multiplier),
            add_image_special_tokens=bool(cfg.model.policy.add_image_special_tokens),
            tokenizer_max_length=int(cfg.model.policy.tokenizer_max_length),
            pad_language_to=str(cfg.model.policy.pad_language_to),
            max_state_dim=int(cfg.model.policy.max_state_dim),
            prefix_length=int(cfg.model.policy.prefix_length),
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            chat=ChatConfig(system_prompt=cfg.model.chat.system_prompt),
            action_tokens=action_cfg,
            use_gpu_preprocessing=use_gpu_preprocessing,
            image_size=image_size,
            camera_keys=(
                None if cfg.model.policy.camera_keys is None else tuple(str(k) for k in cfg.model.policy.camera_keys)
            ),
            empty_cameras=int(cfg.model.policy.empty_cameras),
            flow_hidden_dim=int(flow_cfg.flow_hidden_dim),
            flow_steps=int(flow_cfg.flow_steps),
            latent_loss_weight=float(flow_cfg.latent_loss_weight),
            action_loss_weight=float(flow_cfg.action_loss_weight),
            min_period=float(flow_cfg.min_period),
            max_period=float(flow_cfg.max_period),
            time_beta_alpha=float(flow_cfg.time_beta_alpha),
            time_beta_beta=float(flow_cfg.time_beta_beta),
        )

        backend = SmolVLASharedBackend(
            config=SmolVLASharedBackendConfig(**backend_kwargs),
        )
        backend.setup(device=torch.device("cpu"))
    else:
        raise ValueError(f"Unknown model.backend={backend_type!r}")

    if cfg.data.backend == "lerobot_v3":
        normalization_stats = getattr(datamodule, "normalization_stats", None)
    else:
        normalization_stats = OmegaConf.to_container(
            cfg.model.policy.normalization_stats,
            resolve=True,
        )

    module = PolicyLightningModule(
        backend=backend,
        code_provider=stage1_provider,
        backend_mode=backend_mode,
        normalization_stats=normalization_stats,
        optimizer=PolicyOptimizerConfig(
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
            filename="stage2-policy-step{step:06d}",
            verbose=True,
        ),
    ]
    pipeline_cfg = OmegaConf.select(cfg, "training.validation.pipeline")
    if pipeline_cfg and bool(pipeline_cfg.enabled):
        checks_cfg = OmegaConf.to_container(pipeline_cfg.checks, resolve=True)
        buckets_cfg = OmegaConf.to_container(pipeline_cfg.buckets, resolve=True)
        if not isinstance(checks_cfg, dict):
            raise ValueError("training.validation.pipeline.checks must be a dict")
        if buckets_cfg is None:
            buckets_cfg = {}
        if not isinstance(buckets_cfg, dict):
            raise ValueError("training.validation.pipeline.buckets must be a dict")

        if stage1_provider is None:
            for check_name, raw in checks_cfg.items():
                if not isinstance(raw, dict):
                    continue
                check_type = str(raw.get("type", check_name))
                if check_type == "latent_flow_decode":
                    raw["enabled"] = False

        callbacks.append(
            Stage2ValidationPipelineCallback(
                checks_config=checks_cfg,
                bucket_configs=buckets_cfg,
                max_cached_samples=int(pipeline_cfg.max_cached_samples),
                num_fixed_samples=int(pipeline_cfg.num_fixed_samples),
                run_gc_after_validation=bool(pipeline_cfg.run_gc_after_validation),
            )
        )
    train_viz_cfg = cfg.training.train_visualization
    if train_viz_cfg and bool(train_viz_cfg.enabled):
        callbacks.append(
            PolicyTrainSampleVisualizationCallback(
                PolicyTrainSampleVizConfig(
                    enabled=True,
                    num_samples=int(train_viz_cfg.num_samples),
                    every_n_steps=int(train_viz_cfg.every_n_steps),
                    include_freeform_pred=bool(train_viz_cfg.include_freeform_pred),
                    freeform_max_new_tokens=int(train_viz_cfg.freeform_max_new_tokens),
                )
            )
        )
    data_viz_cfg = OmegaConf.select(cfg.training, "data_visualization")
    if data_viz_cfg and bool(data_viz_cfg.enabled):
        callbacks.append(
            DataSampleVisualizationCallback(
                DataSampleVisualizationConfig(
                    enabled=True,
                    every_n_steps=int(data_viz_cfg.every_n_steps),
                    num_samples=int(data_viz_cfg.num_samples),
                    key=str(data_viz_cfg.key),
                    mode="stage2",
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
            action_dim=int(backend_cfg.action_dim),
            action_chunk_size=int(backend_cfg.action_chunk_size),
            code_seq_len=int(backend_cfg.action_tokens.code_seq_len),
            latent_vector_dim=int(backend_cfg.latent_vector_dim),
            tokenizer_max_length=int(backend_cfg.tokenizer_max_length),
            pad_language_to=str(backend_cfg.pad_language_to),
            system_prompt=backend_cfg.chat.system_prompt,
            max_state_dim=int(backend_cfg.max_state_dim),
            camera_keys=(
                None if backend_cfg.camera_keys is None else tuple(str(k) for k in backend_cfg.camera_keys)
            ),
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
            normalization_stats=normalization_stats,
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
