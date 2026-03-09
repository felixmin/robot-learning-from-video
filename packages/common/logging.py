"""
Logging utilities for LAPA project.

Provides consistent logging across all training stages with WandB integration.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


def setup_logger(
    name: str, log_file: Optional[Path] = None, level=logging.INFO
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set propagate to avoid double logging via root logger
    logger.propagate = False

    # Create formatter (reused for all handlers)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler only if it doesn't exist
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler can be added conditionally even if logger already has handlers
    if log_file:
        # Check if a FileHandler for this specific file already exists
        log_file_resolved = Path(log_file).resolve()
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_file_resolved
            for h in logger.handlers
        )
        if not has_file_handler:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def setup_wandb_logger(cfg: DictConfig) -> WandbLogger:
    """
    Setup WandB logger with configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Configured WandbLogger instance

    Note:
        For large configs, consider logging only a subset:
        `wandb_config = OmegaConf.to_container(cfg.logging_to_log, resolve=True)`
    """
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Ensure save_dir is a string (Hydra may provide Path objects)
    save_dir = cfg.logging.save_dir
    if isinstance(save_dir, Path):
        save_dir = str(save_dir)

    wandb_logger = WandbLogger(
        project=cfg.logging.project,
        name=cfg.experiment_name,
        config=wandb_config,
        save_dir=save_dir,
        log_model=cfg.logging.get("log_model", False),
        tags=cfg.logging.get("tags", []),
    )

    return wandb_logger


def log_hyperparameters(logger: logging.Logger, cfg: DictConfig) -> None:
    """
    Log hyperparameters to console.

    Args:
        logger: Logger instance
        cfg: Hydra configuration

    Note:
        In distributed training, wrap this call with `rank_zero_only` decorator
        from pytorch_lightning.utilities.rank_zero to log only from rank 0.
    """
    logger.info("=" * 80)
    logger.info("Hyperparameters:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
