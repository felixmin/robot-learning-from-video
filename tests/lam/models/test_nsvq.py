"""
Test NSVQ (Noise Substitution Vector Quantization) Module

Tests the vector quantization component ported from LAPA.
"""

import pytest
import torch
from lam.models.nsvq import NSVQ


@pytest.fixture
def nsvq_config(device):
    """Default NSVQ configuration for testing (uses device from conftest)."""
    return {
        "dim": 1024,
        "num_embeddings": 8,
        "embedding_dim": 32,
        "device": device,
        "code_seq_len": 4,
        "patch_size": 32,
        "image_size": 256,
    }


@pytest.fixture
def nsvq_module(nsvq_config):
    """Create NSVQ module instance."""
    return NSVQ(**nsvq_config).to(nsvq_config["device"])


def test_nsvq_initialization(nsvq_config):
    """Test NSVQ module initializes correctly."""
    nsvq = NSVQ(**nsvq_config)

    # Check codebook shape
    assert nsvq.codebooks.shape == (
        nsvq_config["num_embeddings"],
        nsvq_config["embedding_dim"],
    )

    # Check codebook is trainable
    assert nsvq.codebooks.requires_grad

    # Check counters initialized
    assert nsvq.codebooks_used.shape == (nsvq_config["num_embeddings"],)
    assert nsvq.codebooks_used.sum() == 0  # Not used yet


def test_nsvq_forward_training(nsvq_module, nsvq_config):
    """Test NSVQ forward pass in training mode."""
    device = nsvq_config["device"]
    batch_size = 2

    # Create dummy input (after spatial/temporal transformer)
    # Shape: [batch, num_patches, dim]
    num_patches = (
        nsvq_config["image_size"] // nsvq_config["patch_size"]
    ) ** 2  # 8*8=64
    input_first = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device
    )
    input_last = torch.randn(batch_size, num_patches, nsvq_config["dim"], device=device)

    # Forward pass
    quantized, perplexity, codebook_usage, indices = nsvq_module(
        input_first, input_last, codebook_training_only=False
    )

    # Check output shapes
    # NSVQ compresses num_patches → code_seq_len tokens
    assert quantized.shape == (
        batch_size,
        nsvq_config["code_seq_len"],
        nsvq_config["dim"],
    )
    assert perplexity.shape == ()  # Scalar
    assert indices.shape == (batch_size, nsvq_config["code_seq_len"])

    # Check indices are in valid range
    assert indices.min() >= 0
    assert indices.max() < nsvq_config["num_embeddings"]

    # Check perplexity is positive
    assert perplexity.item() > 0

    print(f"✓ Forward pass successful")
    print(f"  - Quantized shape: {quantized.shape}")
    print(f"  - Indices: {indices[0].tolist()}")
    print(f"  - Perplexity: {perplexity.item():.4f}")


def test_nsvq_inference_mode(nsvq_module, nsvq_config):
    """Test NSVQ inference (no gradient noise)."""
    device = nsvq_config["device"]
    batch_size = 2

    num_patches = (nsvq_config["image_size"] // nsvq_config["patch_size"]) ** 2
    input_first = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device
    )
    input_last = torch.randn(batch_size, num_patches, nsvq_config["dim"], device=device)

    # Inference mode
    with torch.no_grad():
        quantized, indices = nsvq_module.inference(input_first, input_last)

    # Check shapes
    # NSVQ compresses num_patches → code_seq_len tokens
    assert quantized.shape == (
        batch_size,
        nsvq_config["code_seq_len"],
        nsvq_config["dim"],
    )
    assert indices.shape == (batch_size, nsvq_config["code_seq_len"])

    # Check indices are discrete
    assert indices.dtype == torch.long

    print(f"✓ Inference mode successful")
    print(f"  - Indices: {indices[0].tolist()}")


def test_nsvq_gradient_flow(nsvq_module, nsvq_config):
    """Test gradients flow through NSVQ (critical for training)."""
    device = nsvq_config["device"]
    batch_size = 2

    num_patches = (nsvq_config["image_size"] // nsvq_config["patch_size"]) ** 2
    input_first = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device, requires_grad=True
    )
    input_last = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device, requires_grad=True
    )

    # Forward pass
    quantized, perplexity, codebook_usage, indices = nsvq_module(
        input_first, input_last, codebook_training_only=False
    )

    # Compute dummy loss
    loss = quantized.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert input_first.grad is not None, "Gradients should flow to input_first"
    assert input_last.grad is not None, "Gradients should flow to input_last"
    assert nsvq_module.codebooks.grad is not None, "Gradients should flow to codebook"

    # Check gradient magnitudes are reasonable
    assert input_first.grad.abs().mean() > 1e-6, "Gradient too small"
    assert nsvq_module.codebooks.grad.abs().mean() > 1e-6, "Codebook gradient too small"

    print(f"✓ Gradient flow verified")
    print(f"  - Input grad mean: {input_first.grad.abs().mean():.6f}")
    print(f"  - Codebook grad mean: {nsvq_module.codebooks.grad.abs().mean():.6f}")


def test_nsvq_codebook_usage(nsvq_module, nsvq_config):
    """Test codebook usage tracking."""
    device = nsvq_config["device"]
    batch_size = 4

    num_patches = (nsvq_config["image_size"] // nsvq_config["patch_size"]) ** 2

    # Run multiple forward passes
    for _ in range(5):
        input_first = torch.randn(
            batch_size, num_patches, nsvq_config["dim"], device=device
        )
        input_last = torch.randn(
            batch_size, num_patches, nsvq_config["dim"], device=device
        )

        quantized, perplexity, codebook_usage, indices = nsvq_module(
            input_first, input_last, codebook_training_only=False
        )

    # Check codebook usage updated
    total_usage = nsvq_module.codebooks_used.sum().item()
    assert total_usage > 0, "Codebook should be used"

    # Check at least one code is used
    num_used_codes = (nsvq_module.codebooks_used > 0).sum().item()
    assert num_used_codes > 0, "At least one codebook entry should be used"

    print(f"✓ Codebook usage tracking works")
    print(f"  - Total uses: {total_usage}")
    print(f"  - Unique codes used: {num_used_codes}/{nsvq_config['num_embeddings']}")
    print(f"  - Usage distribution: {nsvq_module.codebooks_used.tolist()}")


def test_nsvq_different_code_seq_lengths(device):
    """Test NSVQ with different code sequence lengths."""

    # Test supported code_seq_len values (based on NSVQ implementation)
    for code_seq_len in [1, 2, 4]:
        nsvq = NSVQ(
            dim=1024,
            num_embeddings=8,
            embedding_dim=32,
            device=device,
            code_seq_len=code_seq_len,
            patch_size=32,
            image_size=256,
        ).to(device)

        batch_size = 2
        num_patches = 64
        input_first = torch.randn(batch_size, num_patches, 1024, device=device)
        input_last = torch.randn(batch_size, num_patches, 1024, device=device)

        quantized, perplexity, codebook_usage, indices = nsvq(
            input_first, input_last, codebook_training_only=False
        )

        # Check indices match expected code length
        assert indices.shape == (batch_size, code_seq_len)

        print(f"✓ code_seq_len={code_seq_len} works, indices shape: {indices.shape}")


def test_nsvq_replace_unused_codebooks(nsvq_module, nsvq_config):
    """Test codebook replacement mechanism."""
    device = nsvq_config["device"]

    # Manually set some codebooks as unused
    nsvq_module.codebooks_used[0] = 100  # Used
    nsvq_module.codebooks_used[1] = 0  # Unused
    nsvq_module.codebooks_used[2] = 50  # Used
    nsvq_module.codebooks_used[3:] = 0  # All unused

    # Store original codebook values
    original_codebook_1 = nsvq_module.codebooks[1].clone()

    # Replace unused codebooks
    nsvq_module.replace_unused_codebooks()

    # Check unused codebook was replaced (should be different)
    assert not torch.allclose(nsvq_module.codebooks[1], original_codebook_1)

    # Check counters reset
    assert nsvq_module.codebooks_used.sum() == 0

    print(f"✓ Codebook replacement works")


def test_nsvq_get_indices_no_counter_update(nsvq_module, nsvq_config):
    """get_indices must return same indices as forward but not update codebooks_used."""
    device = nsvq_config["device"]
    batch_size = 2
    num_patches = (nsvq_config["image_size"] // nsvq_config["patch_size"]) ** 2
    input_first = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device
    )
    input_last = torch.randn(batch_size, num_patches, nsvq_config["dim"], device=device)

    nsvq_module.codebooks_used.zero_()
    with torch.no_grad():
        indices_get = nsvq_module.get_indices(input_first, input_last)

    assert nsvq_module.codebooks_used.sum().item() == 0
    assert indices_get.shape == (batch_size, nsvq_config["code_seq_len"])
    assert indices_get.min() >= 0
    assert indices_get.max() < nsvq_config["num_embeddings"]


def test_nsvq_get_indices_matches_forward_indices(nsvq_module, nsvq_config):
    """get_indices must produce the same codebook assignments as vq.forward."""
    device = nsvq_config["device"]
    batch_size = 2
    num_patches = (nsvq_config["image_size"] // nsvq_config["patch_size"]) ** 2
    input_first = torch.randn(
        batch_size, num_patches, nsvq_config["dim"], device=device
    )
    input_last = torch.randn(batch_size, num_patches, nsvq_config["dim"], device=device)

    with torch.no_grad():
        _, _, _, indices_forward = nsvq_module(
            input_first, input_last, codebook_training_only=False
        )
        indices_get = nsvq_module.get_indices(input_first, input_last)

    assert torch.equal(indices_forward, indices_get)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
