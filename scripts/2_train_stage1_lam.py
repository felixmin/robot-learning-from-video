#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 2: Train Stage-1 LAM (Latent Action Model)

Train the Stage-1 VQ-VAE model to compress frame-to-frame transitions into discrete latent codes.

Usage:
    # Local debug
    python scripts/2_train_stage1_lam.py experiment=lam_debug

    # Full training on LRZ
    sbatch slurm/train.sbatch scripts/2_train_stage1_lam.py experiment=lam_full
"""

import sys
import inspect
from pathlib import Path

# Add packages to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


from common.data_factory import create_datamodule
from common.callbacks import (
    DataSampleVisualizationCallback,
    DataSampleVisualizationConfig,
    DatasetUsageLoggerCallback,
    ProgressLoggerCallback,
)
from common.run_context import setup_run_context
from stage2.callbacks import ThroughputLoggingCallback, ThroughputLoggingConfig
from common.utils import set_seed, count_parameters
from common.unified_logging import setup_wandb_with_unified_paths
from lam import (
    LAMTask,
    EMACallback,
    TrainPreviewBufferCallback,
    ValidationStrategyCallback,
    create_validation_strategies,
)
from lam import load_lam_model_weights_only


class TrustedFullCheckpointIO(TorchCheckpointIO):
    """Compatibility for trusted Lightning checkpoints under PyTorch>=2.6."""

    def load_checkpoint(self, path, map_location=None, weights_only=False):
        load_kwargs = {}
        if map_location is not None:
            load_kwargs["map_location"] = map_location
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        return torch.load(path, **load_kwargs)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.

    Steps:
    1. Setup logging and seed
    2. Initialize data module
    3. Initialize Stage-1 LAM task
    4. Setup Lightning trainer with callbacks
    5. Train the model
    """
    # Force new W&B run per Hydra job (fix for sweeps)
    if wandb.run:
        wandb.finish()

    logger, output_dir = setup_run_context(
        cfg=cfg,
        workspace_root=workspace_root,
        logger_name="stage1.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 1: LAM Training")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Set random seed for reproducibility (required in base config)
    set_seed(int(cfg.seed))
    logger.info(f"✓ Random seed set to {cfg.seed}")

    # Initialize data module
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Data Module")
    logger.info("=" * 80)

    datamodule = create_datamodule(cfg.data)

    logger.info(f"✓ Data backend: {cfg.data.backend}")
    logger.info(f"  - Batch size: {cfg.data.loader.batch_size}")
    logger.info(f"  - Image size: {cfg.data.preprocess.image_size}")

    if cfg.data.backend != "lerobot_v3":
        raise ValueError(
            f"Only data.backend='lerobot_v3' is supported, got {cfg.data.backend!r}"
        )

    # Initialize Stage-1 LAM task
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Stage-1 LAM Task")
    logger.info("=" * 80)

    model_config = cfg.model
    training_config = cfg.training

    task = LAMTask(
        model_config=model_config,
        training_config=training_config,
    )

    num_params = count_parameters(task.model)
    logger.info("✓ Stage-1 LAM model initialized")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Codebook size: {model_config.codebook_size}")
    logger.info(f"  - Code sequence length: {model_config.code_seq_len}")

    # Setup callbacks
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Callbacks")
    logger.info("=" * 80)

    callbacks = []

    progress_bar_cfg = training_config.progress_bar
    if bool(progress_bar_cfg.enabled):
        callbacks.append(
            TQDMProgressBar(
                refresh_rate=int(progress_bar_cfg.refresh_rate),
                leave=bool(progress_bar_cfg.leave),
            )
        )
        logger.info("✓ Progress bar enabled")
        logger.info(f"  - refresh_rate: {int(progress_bar_cfg.refresh_rate)}")
        logger.info(f"  - leave: {bool(progress_bar_cfg.leave)}")
    else:
        logger.info("✓ Progress bar disabled")

    # Checkpointing (save to unified output directory)
    checkpoint_config = training_config.checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    every_n_train_steps = checkpoint_config.every_n_train_steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
        every_n_train_steps=every_n_train_steps,
        # Avoid using metric keys like `val/loss` in the filename template: Lightning's
        # formatting and filesystem sanitization can be version-dependent, and step-based
        # checkpointing may run when no validation metrics are available.
        filename="stage1-lam-step{step:06d}",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"✓ Checkpoint callback added (monitor={checkpoint_config.monitor})")
    logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  - Checkpointing every {every_n_train_steps} steps")

    # Learning rate monitoring requires a Lightning logger and is added after logger setup.

    # Progress logging (for cluster jobs where tqdm doesn't work in log files)
    progress_logger_cfg = training_config.progress_logger
    if bool(progress_logger_cfg.enabled):
        callbacks.append(
            ProgressLoggerCallback(
                log_every_n_steps=int(progress_logger_cfg.log_every_n_steps)
            )
        )
        logger.info("✓ Progress logger added")
        logger.info(
            f"  - log_every_n_steps: {int(progress_logger_cfg.log_every_n_steps)}"
        )

    # Throughput logging: logs perf/steps_per_sec and perf/samples_per_sec to wandb
    perf_cfg = training_config.throughput
    if bool(perf_cfg.enabled):
        callbacks.append(
            ThroughputLoggingCallback(
                ThroughputLoggingConfig(
                    enabled=True,
                    log_every_n_steps=int(perf_cfg.log_every_n_steps),
                )
            )
        )
        logger.info("✓ Throughput logger added")

    # Dataset usage logging: print dataset mix consumed between validations.
    usage_cfg = training_config.dataset_usage_logger
    if bool(usage_cfg.enabled):
        callbacks.append(
            DatasetUsageLoggerCallback(
                enabled=True,
                log_on_validation_end=bool(usage_cfg.log_on_validation_end),
                log_every_n_steps=int(usage_cfg.log_every_n_steps),
                key=str(usage_cfg.key),
                top_k=int(usage_cfg.top_k),
            )
        )
        logger.info("✓ Dataset usage logger added")

    # Train preview buffer: snapshots from already-consumed train batches.
    # Used by basic visualization to avoid iterating train_dataloader in validation.
    preview_cfg = training_config.train_preview_buffer
    if bool(preview_cfg.enabled):
        callbacks.append(
            TrainPreviewBufferCallback(
                enabled=True,
                max_samples=int(preview_cfg.max_samples),
                samples_per_batch=int(preview_cfg.samples_per_batch),
            )
        )
        logger.info("✓ Train preview buffer callback added")
        logger.info(f"  - max_samples: {int(preview_cfg.max_samples)}")
        logger.info(f"  - samples_per_batch: {int(preview_cfg.samples_per_batch)}")
    else:
        logger.info("✓ Train preview buffer disabled")

    # Lightweight data preview: logs raw train samples without model forward pass.
    data_viz_cfg = OmegaConf.select(training_config, "data_visualization")
    if data_viz_cfg and bool(data_viz_cfg.enabled):
        callbacks.append(
            DataSampleVisualizationCallback(
                DataSampleVisualizationConfig(
                    enabled=True,
                    every_n_steps=int(data_viz_cfg.every_n_steps),
                    num_samples=int(data_viz_cfg.num_samples),
                    key=str(data_viz_cfg.key),
                    mode="stage1",
                )
            )
        )
        logger.info("✓ Data sample visualization callback added")
        logger.info(f"  - every_n_steps: {int(data_viz_cfg.every_n_steps)}")
        logger.info(f"  - num_samples: {int(data_viz_cfg.num_samples)}")

    # Setup validation strategies
    val_config = cfg.validation
    strategies_config = val_config.strategies
    if strategies_config:
        strategies_config = OmegaConf.to_container(strategies_config, resolve=True)

    bucket_configs = OmegaConf.to_container(val_config.buckets, resolve=True)

    bucket_filters = None
    if bucket_configs is not None:
        bucket_filters = {name: cfg["filters"] for name, cfg in bucket_configs.items()}

    strategies = create_validation_strategies(
        strategies_config, bucket_filters=bucket_filters
    )

    val_strategy_callback = ValidationStrategyCallback(
        strategies=strategies,
        bucket_configs=bucket_configs,  # Pass bucket configs for routing
        num_fixed_samples=val_config.num_fixed_samples,
        num_random_samples=val_config.num_random_samples,
        max_cached_samples=val_config.max_cached_samples,
    )
    callbacks.append(val_strategy_callback)
    logger.info(f"✓ Validation strategy callback added ({len(strategies)} strategies)")
    logger.info(f"  - Max cached samples: {val_config.max_cached_samples}")
    if bucket_configs:
        logger.info(f"  - Buckets: {list(bucket_configs.keys())}")
    for strategy in strategies:
        bucket_info = f", buckets={strategy.buckets}" if strategy.buckets else ""
        logger.info(
            f"  - {strategy.name}: every {strategy.every_n_validations} validations{bucket_info}"
        )

    # Optional EMA
    if training_config.use_ema:
        ema_callback = EMACallback(
            decay=training_config.ema_decay,
            update_every=training_config.ema_update_every,
            update_after_step=training_config.ema_update_after_step,
        )
        callbacks.append(ema_callback)
        logger.info("✓ EMA callback added")

    # Setup WandB logger (use unified logging paths)
    logger.info("\n" + "=" * 80)
    logger.info("Setting up WandB Logger")
    logger.info("=" * 80)

    if cfg.logging.use_wandb:
        # Calculate unique run name for sweeps
        run_name = cfg.experiment.name
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == "MULTIRUN":
            run_name = f"{run_name}_{hydra_cfg.job.num}"

        wandb_logger = setup_wandb_with_unified_paths(
            logger=logger,
            output_dir=output_dir,
            project=cfg.logging.project,
            name=run_name,
            group=cfg.experiment.name,
            tags=cfg.logging.tags,
            use_wandb=True,
        )
    else:
        wandb_logger = None
        logger.info("✓ WandB disabled")

    # Learning rate monitoring (requires a Lightning logger)
    if wandb_logger is not None:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        logger.info("✓ Learning rate monitor added")
    else:
        logger.info("✓ Learning rate monitor disabled (no logger)")

    # Setup profiler
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Profiler")
    logger.info("=" * 80)

    profiler = None
    profiler_type = None
    if training_config.profiler.enabled:
        profiler_type = training_config.profiler.type
        dirpath = str(output_dir / "profiles")
        filename = training_config.profiler.filename

        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler

            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
            logger.info("✓ SimpleProfiler enabled")
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler

            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
            logger.info("✓ AdvancedProfiler enabled")
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
            logger.info("✓ PyTorchProfiler enabled (high overhead!)")
        logger.info(f"  - Output: {dirpath}/{filename}")
    else:
        logger.info("✓ Profiler disabled")

    # Setup trainer
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Trainer")
    logger.info("=" * 80)

    # Get validation check interval from config
    val_check_interval = val_config.check_interval
    limit_val_batches = val_config.limit_batches
    check_val_every_n_epoch = val_config.check_val_every_n_epoch

    log_every_n_steps = int(training_config.log_every_n_steps)
    num_sanity_val_steps = OmegaConf.select(training_config, "num_sanity_val_steps")
    trainer_strategy = OmegaConf.select(training_config, "strategy", default="auto")
    trainer_devices = OmegaConf.select(training_config, "devices", default="auto")
    trainer_num_nodes = int(OmegaConf.select(training_config, "num_nodes", default=1))

    model_summary_cfg = training_config.model_summary

    ckpt_path = training_config.resume_from_checkpoint
    trainer_plugins = None
    if ckpt_path:
        trainer_plugins = [TrustedFullCheckpointIO()]
        logger.info("✓ Resume compatibility: forcing full checkpoint deserialization")

    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        max_steps=int(training_config.max_steps),
        accelerator="auto",
        devices=trainer_devices,
        num_nodes=trainer_num_nodes,
        strategy=trainer_strategy,
        plugins=trainer_plugins,
        precision=cfg.precision,
        gradient_clip_val=training_config.gradient.clip_val,
        gradient_clip_algorithm=training_config.gradient.clip_algorithm,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else False,
        default_root_dir=str(output_dir),
        profiler=profiler,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,  # Configurable validation frequency
        limit_val_batches=limit_val_batches,  # Limit validation batches
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=(
            int(num_sanity_val_steps) if num_sanity_val_steps is not None else None
        ),
        enable_progress_bar=bool(progress_bar_cfg.enabled),
        enable_model_summary=bool(model_summary_cfg.enabled),
    )

    logger.info("✓ Trainer initialized")
    logger.info(f"  - Max epochs: {training_config.epochs}")
    logger.info(f"  - Val check interval: {val_check_interval}")
    logger.info(f"  - Limit val batches: {limit_val_batches}")
    logger.info(f"  - Check val every n epoch: {check_val_every_n_epoch}")
    logger.info(f"  - Num sanity val steps: {num_sanity_val_steps}")
    logger.info(f"  - Precision: {cfg.precision}")
    logger.info("  - Accelerator: auto")
    logger.info(f"  - Devices: {trainer_devices}")
    logger.info(f"  - Num nodes: {trainer_num_nodes}")
    logger.info(f"  - Strategy: {trainer_strategy}")
    logger.info(f"  - Profiler: {profiler_type if profiler else 'disabled'}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    # Check for weight-only loading (new scheduler, fresh optimizer)
    weights_path = training_config.load_weights_from
    if weights_path:
        logger.info(f"✓ Loading model weights from: {weights_path}")
        missing, unexpected, loaded_count = load_lam_model_weights_only(
            task.model, weights_path, strict=False
        )
        if missing:
            logger.warning(
                f"  - Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            logger.warning(
                f"  - Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        logger.info(
            f"  - Loaded {loaded_count} weight tensors (fresh optimizer/scheduler)"
        )

    # Check for full resume (restores optimizer/scheduler state)
    if ckpt_path:
        logger.info(f"✓ Resuming from checkpoint: {ckpt_path}")

    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    # Print best checkpoint
    if checkpoint_callback.best_model_path:
        logger.info(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        best_score = checkpoint_callback.best_model_score
        if best_score is None:
            logger.warning(
                "  - Best val/loss unavailable (no validation metric recorded in this run state)"
            )
        else:
            logger.info(f"  - Best val/loss: {float(best_score):.4f}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
