#!/usr/bin/env python3
"""
Unified logging setup for cluster and local runs.

This module provides:
1. Consolidated logging to runs/ folder
2. SLURM job ID integration
3. Stdout/stderr capture to log files
4. WandB integration with proper paths
5. Hydra output directory configuration
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional
import contextlib


def get_job_id() -> str:
    """
    Get the job identifier (SLURM_JOB_ID or wandb run ID).

    Priority:
    1. SLURM_JOB_ID (cluster runs)
    2. WANDB_RUN_ID (if WandB is initialized)
    3. 'local' (local development)

    Returns:
        Job identifier string
    """
    # Check for SLURM job ID first
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return job_id

    # Check for WandB run ID (set during wandb.init())
    wandb_id = os.environ.get("WANDB_RUN_ID")
    if wandb_id:
        return wandb_id

    # Default to 'local' for development
    return "local"


def _make_excepthook(original_hook):
    """
    Create an exception hook that logs uncaught exceptions before calling the original hook.

    This ensures crashes are captured in the log file instead of only appearing in stderr.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        # Don't log KeyboardInterrupt (Ctrl+C)
        if issubclass(exc_type, KeyboardInterrupt):
            original_hook(exc_type, exc_value, exc_traceback)
            return
        logging.critical(
            "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
        )
        original_hook(exc_type, exc_value, exc_traceback)

    return handle_exception


def get_rank() -> int:
    """
    Get the current process rank for distributed training.

    Checks environment variables set by PyTorch DDP, torchrun, and SLURM.

    Returns:
        Process rank (0 for main process or single-GPU training)
    """
    # PyTorch DDP / torchrun
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # SLURM
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    # Single process
    return 0


def _slugify_run_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "run"


def resolve_runs_dir(
    logging_root_dir: Optional[str],
    logging_runs_dir: Optional[str],
    workspace_root: Path,
    experiment_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Resolve the runs directory from logging config values.

    Priority:
    1. logging_runs_dir (explicit run directory, e.g., runs/2026-01-15_191650_lam_debug)
    2. <logging_root_dir>/runs/<timestamp>_<experiment> (if logging_root_dir is set)
    3. <workspace_root>/runs/<timestamp>_<experiment> (fallback to project root)

    Args:
        logging_root_dir: Optional base directory for all run artifacts (cfg.logging.root_dir)
        logging_runs_dir: Optional explicit run directory (cfg.logging.runs_dir)
        workspace_root: Project root directory as fallback
        experiment_name: Optional experiment name used to name the run folder
        timestamp: Optional timestamp string override (primarily for tests)

    Returns:
        Resolved Path to the runs directory (flat structure)
    """
    if logging_runs_dir:
        runs_dir = Path(logging_runs_dir)
        if (not runs_dir.is_absolute()) and logging_root_dir:
            runs_dir = Path(logging_root_dir) / runs_dir
        return runs_dir

    base_root = Path(logging_root_dir) if logging_root_dir else workspace_root
    safe_experiment = _slugify_run_name(experiment_name) if experiment_name else "local"
    timestamp = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    return base_root / "runs" / f"{timestamp}_{safe_experiment}"


def setup_unified_logging(
    runs_dir: Path,
    job_id: Optional[str] = None,
    log_level: str = "INFO",
    logger_name: str = "hlrp.training",
) -> tuple[logging.Logger, Path]:
    """
    Setup unified logging that captures all output to a single run directory.

    Creates:
    - <runs_dir>/unified.log - Complete training log (rank 0 only)
    - <runs_dir>/ - Output directory for checkpoints, wandb, etc.

    Args:
        runs_dir: Path to the run directory (flat structure)
        job_id: Optional job ID (used for logging, auto-detected if None)
        log_level: Logging level (INFO, DEBUG, etc.)
        logger_name: Name for the returned logger (default: "hlrp.training")

    Returns:
        (logger, output_dir) tuple

    Note:
        In distributed training, only rank 0 writes to the log file to prevent
        interleaved output. All ranks still log to console.
    """
    if job_id is None:
        job_id = get_job_id()

    # Flat structure: output directly to runs_dir
    output_dir = runs_dir

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Failed to create directories: {e}")
        print(f"  - Attempted to create: {output_dir}")
        raise

    # Log file path
    log_file = output_dir / "unified.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (log file) - only rank 0 writes to file in distributed training
    rank = get_rank()
    if rank == 0:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            print(f"WARNING: Failed to create log file handler: {e}")
            print(f"  - Attempted to write to: {log_file}")
            print("  - Logging will continue to console only")

        # Install excepthook to capture uncaught exceptions in log file
        sys.excepthook = _make_excepthook(sys.excepthook)

    # Get module-specific logger
    logger = logging.getLogger(logger_name)

    # Log setup info
    logger.info("=" * 80)
    logger.info("Unified Logging Initialized")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log level: {log_level}")
    logger.info("=" * 80)

    # Note: Hydra creates its own output directory for config backups (hydra.run.dir).
    # Our unified logging handles the important outputs under <runs_dir>/:
    #   - Checkpoints → checkpoints/
    #   - WandB → wandb/
    #   - Logs → unified.log
    #   - Hydra config → .hydra/

    # Note: stdout/stderr capture is handled by WandB when enabled
    # Our file handler above captures all logging.* calls
    # WandB's stdout wrapper captures all print() calls
    # This creates two complementary logs:
    #   - <runs_dir>/unified.log: logger.info() calls (timestamped)
    #   - <runs_dir>/wandb/files/output.log: print() calls (WandB capture)

    return logger, output_dir


def setup_wandb_with_unified_paths(
    logger: logging.Logger,
    output_dir: Path,
    project: str,
    name: str,
    tags: list,
    use_wandb: bool = True,
    **kwargs,
):
    """
    Setup WandB logger with paths integrated into unified logging structure.

    Args:
        logger: Logger instance
        output_dir: Output directory from setup_unified_logging
        project: WandB project name
        name: Run name
        tags: WandB tags
        use_wandb: Whether to enable WandB
        **kwargs: Additional arguments passed to WandbLogger

    Returns:
        WandbLogger or None
    """
    if not use_wandb:
        logger.info("WandB disabled")
        return None

    from lightning.pytorch.loggers import WandbLogger

    # WandB saves to output_dir/wandb
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)

    wandb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir=str(wandb_dir),
        tags=tags,
        **kwargs,
    )

    logger.info(f"✓ WandB logger initialized (project={project})")
    logger.info(f"  - WandB directory: {wandb_dir}")

    return wandb_logger


@contextlib.contextmanager
def logging_context(
    workspace_root: Path, job_id: Optional[str] = None, log_level: str = "INFO"
):
    """
    Context manager for unified logging setup.

    Usage:
        with logging_context(workspace_root) as (logger, output_dir):
            logger.info("Training started")
            # ... training code ...
    """
    logger, output_dir = setup_unified_logging(
        runs_dir=workspace_root,
        job_id=job_id,
        log_level=log_level,
    )

    try:
        yield logger, output_dir
    finally:
        # Cleanup: flush and close handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
