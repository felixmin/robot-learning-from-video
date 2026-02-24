#!/usr/bin/env python3
"""
Script 2: Train LAQ (Latent Action Quantization)

Train the VQ-VAE model to compress frame-to-frame transitions into discrete latent codes.

Usage:
    # Local debug
    python scripts/2_train_laq.py experiment=laq_debug

    # Full training on LRZ
    sbatch slurm/train.sbatch scripts/2_train_laq.py experiment=laq_full
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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, TQDMProgressBar
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


from common.data_factory import create_datamodule
from common.callbacks import DatasetUsageLoggerCallback, ProgressLoggerCallback
from common.cache_env import configure_cache_env, resolve_cache_dir
from foundation.callbacks import ThroughputLoggingCallback, ThroughputLoggingConfig
from common.logging import set_seed, count_parameters
from common.unified_logging import resolve_runs_dir, setup_unified_logging, setup_wandb_with_unified_paths
from laq import (
    LAQTask,
    EMACallback,
    TrainPreviewBufferCallback,
    ValidationStrategyCallback,
    create_validation_strategies,
)


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
    3. Initialize LAQ task
    4. Setup Lightning trainer with callbacks
    5. Train the model
    """
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
            logging_root_dir=cfg.logging.get("root_dir"),
            logging_runs_dir=cfg.logging.get("runs_dir"),
            workspace_root=workspace_root,
            experiment_name=OmegaConf.select(cfg, "experiment.name"),
        )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.get("job_id"),
        log_level=cfg.logging.get("level", "INFO"),
        logger_name="laq.training",
    )

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    logger.info("=" * 80)
    logger.info("LAPA Stage 1: LAQ Training")
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
    datamodule.setup()

    logger.info(f"✓ Data backend: {cfg.data.backend}")
    logger.info(f"  - Batch size: {cfg.data.loader.batch_size}")
    logger.info(f"  - Image size: {cfg.data.preprocess.image_size}")

    if cfg.data.backend == "oxe_local_indexed":
        dataset_names = [d.name for d in cfg.data.dataset.oxe.datasets]
        logger.info(f"  - Datasets: {dataset_names}")
        try:
            est_batches = int(len(datamodule.train_dataset))
            est_pairs = est_batches * int(cfg.data.loader.batch_size)
            logger.info(
                f"  - Estimated train batches/epoch: ~{est_batches:,} (~{est_pairs:,} pairs)"
            )
        except TypeError:
            logger.info("  - Train dataset length is not available")
    else:
        raise ValueError(
            f"Only data.backend='oxe_local_indexed' is supported, got {cfg.data.backend!r}"
        )

    # Initialize LAQ task
    logger.info("\n" + "=" * 80)
    logger.info("Initializing LAQ Task")
    logger.info("=" * 80)

    model_config = cfg.model
    training_config = cfg.training

    task = LAQTask(
        model_config=model_config,
        training_config=training_config,
        use_ema=training_config.get("use_ema", False),
    )

    num_params = count_parameters(task.model)
    logger.info(f"✓ LAQ model initialized")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Codebook size: {model_config.codebook_size}")
    logger.info(f"  - Code sequence length: {model_config.code_seq_len}")

    # Setup callbacks
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Callbacks")
    logger.info("=" * 80)

    callbacks = []

    progress_bar_cfg = training_config.get("progress_bar")
    if progress_bar_cfg is None:
        raise ValueError(
            "Missing `training.progress_bar` config. Expected:\n"
            "training:\n"
            "  progress_bar:\n"
            "    enabled: true|false\n"
            "    refresh_rate: <int>\n"
            "    leave: true|false\n"
        )
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
        filename="laq-step{step:06d}",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"✓ Checkpoint callback added (monitor={checkpoint_config.monitor})")
    logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  - Checkpointing every {every_n_train_steps} steps")

    # Learning rate monitoring requires a Lightning logger and is added after logger setup.

    # Progress logging (for cluster jobs where tqdm doesn't work in log files)
    progress_logger_cfg = training_config.get("progress_logger")
    if progress_logger_cfg is None:
        raise ValueError(
            "Missing `training.progress_logger` config. Expected:\n"
            "training:\n"
            "  progress_logger:\n"
            "    enabled: true|false\n"
            "    log_every_n_steps: <int>\n"
        )
    if bool(progress_logger_cfg.enabled):
        callbacks.append(
            ProgressLoggerCallback(
                log_every_n_steps=int(progress_logger_cfg.log_every_n_steps)
            )
        )
        logger.info("✓ Progress logger added")
        logger.info(f"  - log_every_n_steps: {int(progress_logger_cfg.log_every_n_steps)}")

    # Throughput logging: logs perf/steps_per_sec and perf/samples_per_sec to wandb
    perf_cfg = training_config.get("throughput")
    if perf_cfg and bool(perf_cfg.get("enabled", True)):
        if perf_cfg.get("log_every_n_steps") is None:
            raise ValueError(
                "Missing `training.throughput.log_every_n_steps` config. Expected:\n"
                "training:\n"
                "  throughput:\n"
                "    enabled: true\n"
                "    log_every_n_steps: <int>\n"
            )
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
    usage_cfg = training_config.get("dataset_usage_logger")
    if usage_cfg and bool(usage_cfg.get("enabled", True)):
        if usage_cfg.get("log_every_n_steps") is None:
            raise ValueError(
                "Missing `training.dataset_usage_logger.log_every_n_steps` config. Expected:\n"
                "training:\n"
                "  dataset_usage_logger:\n"
                "    enabled: true\n"
                "    log_every_n_steps: <int>\n"
            )
        callbacks.append(
            DatasetUsageLoggerCallback(
                enabled=True,
                log_on_validation_end=bool(usage_cfg.get("log_on_validation_end", True)),
                log_every_n_steps=int(usage_cfg.log_every_n_steps),
                key=str(usage_cfg.get("key", "dataset_name")),
                top_k=int(usage_cfg.get("top_k", 12)),
            )
        )
        logger.info("✓ Dataset usage logger added")

    # Train preview buffer: snapshots from already-consumed train batches.
    # Used by basic visualization to avoid iterating train_dataloader in validation.
    preview_cfg = training_config.get("train_preview_buffer")
    if preview_cfg is None:
        raise ValueError(
            "Missing `training.train_preview_buffer` config. Expected:\n"
            "training:\n"
            "  train_preview_buffer:\n"
            "    enabled: true|false\n"
            "    max_samples: <int>\n"
            "    samples_per_batch: <int>\n"
        )
    if bool(preview_cfg.get("enabled", True)):
        if preview_cfg.get("max_samples") is None:
            raise ValueError("Missing `training.train_preview_buffer.max_samples` config.")
        if preview_cfg.get("samples_per_batch") is None:
            raise ValueError("Missing `training.train_preview_buffer.samples_per_batch` config.")
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

    # Setup validation strategies
    val_config = cfg.get("validation")
    if val_config is None:
        raise ValueError(
            "Missing top-level `validation` config. "
            "Compose a validation preset via `/validation@validation: ...`."
        )
    strategies_config = val_config.get("strategies", {})
    if strategies_config:
        strategies_config = OmegaConf.to_container(strategies_config, resolve=True)

    bucket_configs = val_config.get("buckets")
    if bucket_configs is not None:
        bucket_configs = OmegaConf.to_container(bucket_configs, resolve=True)

    bucket_filters = None
    if bucket_configs is not None:
        bucket_filters = {name: cfg["filters"] for name, cfg in bucket_configs.items()}

    strategies = create_validation_strategies(strategies_config, bucket_filters=bucket_filters)

    val_strategy_callback = ValidationStrategyCallback(
        strategies=strategies,
        bucket_configs=bucket_configs,  # Pass bucket configs for routing
        num_fixed_samples=val_config.get("num_fixed_samples", 8),
        num_random_samples=val_config.get("num_random_samples", 8),
        max_cached_samples=val_config.get("max_cached_samples", 256),
    )
    callbacks.append(val_strategy_callback)
    logger.info(f"✓ Validation strategy callback added ({len(strategies)} strategies)")
    logger.info(f"  - Max cached samples: {val_config.get('max_cached_samples', 256)}")
    if bucket_configs:
        logger.info(f"  - Buckets: {list(bucket_configs.keys())}")
    for strategy in strategies:
        bucket_info = f", buckets={strategy.buckets}" if strategy.buckets else ""
        logger.info(f"  - {strategy.name}: every {strategy.every_n_validations} validations{bucket_info}")

    # Optional EMA
    if training_config.get("use_ema", False):
        ema_callback = EMACallback(
            decay=training_config.get("ema_decay", 0.999),
            update_every=training_config.get("ema_update_every", 1),
            update_after_step=training_config.get("ema_update_after_step", 0),
        )
        callbacks.append(ema_callback)
        logger.info("✓ EMA callback added")

    # Setup WandB logger (use unified logging paths)
    logger.info("\n" + "=" * 80)
    logger.info("Setting up WandB Logger")
    logger.info("=" * 80)

    if hasattr(cfg, "logging") and cfg.logging.get("use_wandb", True):
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
            project=cfg.logging.get("project", "hlrp"),
            name=run_name,
            group=cfg.experiment.name,
            tags=cfg.logging.get("tags", []),
            use_wandb=True,
            settings=wandb.Settings(start_method="thread", reinit=True),
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
    if training_config.get("profiler", {}).get("enabled", False):
        profiler_type = training_config.profiler.get("type", "simple")
        dirpath = str(output_dir / "profiles")
        filename = training_config.profiler.get("filename", "profile")

        if profiler_type == "simple":
            from lightning.pytorch.profilers import SimpleProfiler
            profiler = SimpleProfiler(dirpath=dirpath, filename=filename)
            logger.info(f"✓ SimpleProfiler enabled")
        elif profiler_type == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler
            profiler = AdvancedProfiler(dirpath=dirpath, filename=filename)
            logger.info(f"✓ AdvancedProfiler enabled")
        elif profiler_type == "pytorch":
            from lightning.pytorch.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=dirpath,
                filename=filename,
                emit_nvtx=False,
                export_to_chrome=True,
                row_limit=20,
            )
            logger.info(f"✓ PyTorchProfiler enabled (high overhead!)")
        logger.info(f"  - Output: {dirpath}/{filename}")
    else:
        logger.info("✓ Profiler disabled")

    # Setup trainer
    logger.info("\n" + "=" * 80)
    logger.info("Setting up Trainer")
    logger.info("=" * 80)

    # Get validation check interval from config
    if val_config.get("check_interval") is None:
        raise ValueError("Missing `validation.check_interval` config.")
    if val_config.get("limit_batches") is None:
        raise ValueError("Missing `validation.limit_batches` config.")
    val_check_interval = val_config.check_interval
    limit_val_batches = val_config.limit_batches
    check_val_every_n_epoch = val_config.get("check_val_every_n_epoch", 1)

    if training_config.get("log_every_n_steps") is None:
        raise ValueError("Missing `training.log_every_n_steps` config.")
    log_every_n_steps = int(training_config.log_every_n_steps)

    model_summary_cfg = training_config.get("model_summary")
    if model_summary_cfg is None:
        raise ValueError(
            "Missing `training.model_summary` config. Expected:\n"
            "training:\n"
            "  model_summary:\n"
            "    enabled: true|false\n"
        )

    ckpt_path = training_config.get("resume_from_checkpoint", None)
    trainer_plugins = None
    if ckpt_path:
        trainer_plugins = [TrustedFullCheckpointIO()]
        logger.info("✓ Resume compatibility: forcing full checkpoint deserialization")

    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        max_steps=training_config.get("max_steps") or -1,  # Convert None to -1
        accelerator="auto",
        devices="auto",
        strategy="auto",
        plugins=trainer_plugins,
        precision=cfg.get("precision", "32-true"),
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
        enable_progress_bar=bool(progress_bar_cfg.enabled),
        enable_model_summary=bool(model_summary_cfg.enabled),
    )

    logger.info(f"✓ Trainer initialized")
    logger.info(f"  - Max epochs: {training_config.epochs}")
    logger.info(f"  - Val check interval: {val_check_interval}")
    logger.info(f"  - Limit val batches: {limit_val_batches}")
    logger.info(f"  - Check val every n epoch: {check_val_every_n_epoch}")
    logger.info(f"  - Precision: {cfg.get('precision', '32-true')}")
    logger.info(f"  - Accelerator: auto")
    logger.info(f"  - Devices: auto")
    logger.info(f"  - Profiler: {profiler_type if profiler else 'disabled'}")

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    # Check for weight-only loading (new scheduler, fresh optimizer)
    weights_path = training_config.get("load_weights_from", None)
    if weights_path:
        logger.info(f"✓ Loading model weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Load only model state_dict, not optimizer/scheduler
        state_dict = ckpt.get("state_dict", ckpt)
        # Filter to model weights only (remove 'model.' prefix if present from Lightning)
        model_state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
                       for k, v in state_dict.items() if not k.startswith(("optimizer", "lr_scheduler"))}
        missing, unexpected = task.model.load_state_dict(model_state, strict=False)
        if missing:
            logger.warning(f"  - Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"  - Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        logger.info(f"  - Loaded {len(model_state)} weight tensors (fresh optimizer/scheduler)")

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
            logger.warning("  - Best val/loss unavailable (no validation metric recorded in this run state)")
        else:
            logger.info(f"  - Best val/loss: {float(best_score):.4f}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
