"""
Shared pytest fixtures and configuration for all tests.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add packages to path so all tests can import modules
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))


@pytest.fixture(scope="session")
def device():
    """Get device for testing (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def workspace_root_path():
    """Path to workspace root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_path(workspace_root_path):
    """Path to test data directory."""
    return workspace_root_path / "tests" / "data"


# Default configurations for LAM components
@pytest.fixture
def lam_config():
    """Standard LAM model configuration for testing."""
    return {
        "dim": 1024,
        "quant_dim": 32,
        "codebook_size": 8,
        "image_size": 256,
        "patch_size": 32,
        "spatial_depth": 8,
        "temporal_depth": 8,
        "dim_head": 64,
        "heads": 16,
        "code_seq_len": 4,
    }


@pytest.fixture
def small_lam_config():
    """Smaller LAM config for faster tests."""
    return {
        "dim": 256,
        "quant_dim": 16,
        "codebook_size": 8,
        "image_size": 256,
        "patch_size": 32,
        "spatial_depth": 2,
        "temporal_depth": 2,
        "dim_head": 32,
        "heads": 4,
        "code_seq_len": 4,
    }
