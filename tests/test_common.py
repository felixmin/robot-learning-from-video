"""
Tests for common utilities.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from common.utils import set_seed, count_parameters


def test_set_seed():
    """Test that seed setting produces reproducible results."""
    set_seed(42)
    x1 = torch.randn(10)

    set_seed(42)
    x2 = torch.randn(10)

    assert torch.allclose(x1, x2), "Seeds should produce identical random numbers"


def test_count_parameters():
    """Test parameter counting."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),  # 10*20 + 20 = 220
        torch.nn.Linear(20, 5),  # 20*5 + 5 = 105
    )

    total = count_parameters(model, trainable_only=False)
    assert total == 325, f"Expected 325 parameters, got {total}"

    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False

    trainable = count_parameters(model, trainable_only=True)
    assert trainable == 105, f"Expected 105 trainable parameters, got {trainable}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
