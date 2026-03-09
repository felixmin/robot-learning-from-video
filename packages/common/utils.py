"""
General utility functions for LAPA project.
"""

import random

import numpy as np
import torch


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
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module, logger=None) -> None:
    """
    Print model summary with parameter counts.

    Args:
        model: PyTorch model
        logger: Optional logger instance
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    summary = (
        f"\nModel Summary:\n"
        f"  Total parameters: {total_params:,}\n"
        f"  Trainable parameters: {trainable_params:,}\n"
        f"  Non-trainable parameters: {total_params - trainable_params:,}\n"
    )

    if logger:
        logger.info(summary)
    else:
        print(summary)
