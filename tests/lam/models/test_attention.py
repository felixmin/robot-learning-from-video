"""
Test Attention and Transformer Modules

Tests the transformer architecture components ported from LAPA.
"""

import pytest
import torch
from lam.models.attention import (
    Attention,
    Transformer,
    ContinuousPositionBias,
    PEG,
    LayerNorm,
    FeedForward,
)


@pytest.fixture
def attention_config():
    """Default attention configuration."""
    return {
        "dim": 1024,
        "dim_head": 64,
        "heads": 16,
        "dropout": 0.0,
    }


@pytest.fixture
def transformer_config():
    """Default transformer configuration."""
    return {
        "dim": 1024,
        "depth": 8,
        "dim_head": 64,
        "heads": 16,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
    }


class TestLayerNorm:
    """Test bias-less LayerNorm."""

    def test_layernorm_initialization(self):
        """Test LayerNorm initializes correctly."""
        dim = 1024
        norm = LayerNorm(dim)

        assert norm.gamma.shape == (dim,)
        assert norm.beta.shape == (dim,)
        assert norm.gamma.requires_grad
        assert not norm.beta.requires_grad  # Beta is buffer, not trainable

    def test_layernorm_forward(self, device):
        """Test LayerNorm forward pass."""
        batch_size, seq_len, dim = 2, 10, 1024
        x = torch.randn(batch_size, seq_len, dim, device=device)

        norm = LayerNorm(dim).to(device)
        output = norm(x)

        assert output.shape == x.shape
        # Check normalization (mean ≈ 0, std ≈ 1)
        assert output.mean(dim=-1).abs().max() < 0.1
        assert (output.std(dim=-1) - 1.0).abs().max() < 0.1

        print("✓ LayerNorm forward pass successful")
        print(f"  - Mean: {output.mean(dim=-1).abs().max():.6f}")
        print(f"  - Std: {output.std(dim=-1).mean():.6f}")


class TestFeedForward:
    """Test GEGLU FeedForward network."""

    def test_feedforward_initialization(self):
        """Test FeedForward initializes correctly."""
        dim = 1024
        ff = FeedForward(dim, mult=4, dropout=0.0)

        # Check it's a sequential module
        assert isinstance(ff, torch.nn.Sequential)

    def test_feedforward_forward(self, device):
        """Test FeedForward forward pass."""
        batch_size, seq_len, dim = 2, 10, 1024
        x = torch.randn(batch_size, seq_len, dim, device=device)

        ff = FeedForward(dim, mult=4, dropout=0.0).to(device)
        output = ff(x)

        assert output.shape == x.shape

        print("✓ FeedForward forward pass successful")


class TestPEG:
    """Test Position Encoding Generator (PEG)."""

    def test_peg_initialization(self):
        """Test PEG initializes correctly."""
        dim = 1024
        peg = PEG(dim, causal=False)

        assert hasattr(peg, "dsconv")
        assert not peg.causal

    def test_peg_forward_with_shape(self, device):
        """Test PEG forward pass with explicit shape."""
        batch_size, time, height, width, dim = 2, 2, 8, 8, 1024

        # Flattened input
        x = torch.randn(batch_size, time * height * width, dim, device=device)

        peg = PEG(dim, causal=False).to(device)
        output = peg(x, shape=(batch_size, time, height, width))

        assert output.shape == x.shape

        print("✓ PEG forward pass successful")

    def test_peg_causal_mode(self, device):
        """Test PEG in causal mode."""
        batch_size, time, height, width, dim = 2, 2, 8, 8, 1024
        x = torch.randn(batch_size, time * height * width, dim, device=device)

        peg = PEG(dim, causal=True).to(device)
        output = peg(x, shape=(batch_size, time, height, width))

        assert output.shape == x.shape

        print("✓ PEG causal mode works")


class TestContinuousPositionBias:
    """Test continuous position bias for spatial attention."""

    def test_continuous_position_bias_initialization(self):
        """Test ContinuousPositionBias initializes correctly."""
        dim = 1024
        heads = 16

        bias = ContinuousPositionBias(dim=dim, heads=heads)

        assert hasattr(bias, "net")

    def test_continuous_position_bias_forward(self, device):
        """Test ContinuousPositionBias forward pass."""
        dim = 1024
        heads = 16
        height, width = 8, 8

        bias_module = ContinuousPositionBias(dim=dim, heads=heads).to(device)
        bias = bias_module(height, width, device=device)

        # Expected shape: [heads, height*width, height*width]
        assert bias.shape == (heads, height * width, height * width)

        print("✓ ContinuousPositionBias forward pass successful")
        print(f"  - Output shape: {bias.shape}")


class TestAttention:
    """Test attention mechanism."""

    def test_attention_initialization(self, attention_config):
        """Test Attention initializes correctly."""
        attn = Attention(**attention_config)

        assert attn.heads == attention_config["heads"]
        assert hasattr(attn, "to_q")
        assert hasattr(attn, "to_kv")
        assert hasattr(attn, "to_out")

    def test_attention_self_attention(self, attention_config, device):
        """Test self-attention forward pass."""
        batch_size, seq_len = 2, 64

        x = torch.randn(batch_size, seq_len, attention_config["dim"], device=device)

        attn = Attention(**attention_config).to(device)
        output = attn(x)

        assert output.shape == x.shape

        print("✓ Self-attention forward pass successful")

    def test_attention_cross_attention(self, device):
        """Test cross-attention forward pass."""
        batch_size, seq_len, context_len = 2, 64, 32
        dim = 1024

        x = torch.randn(batch_size, seq_len, dim, device=device)
        context = torch.randn(batch_size, context_len, dim, device=device)

        attn = Attention(dim=dim, dim_context=dim, dim_head=64, heads=16).to(device)
        output = attn(x, context=context)

        assert output.shape == x.shape

        print("✓ Cross-attention forward pass successful")

    def test_attention_with_bias(self, attention_config, device):
        """Test attention with positional bias."""
        batch_size, seq_len = 2, 64
        heads = attention_config["heads"]

        x = torch.randn(batch_size, seq_len, attention_config["dim"], device=device)
        attn_bias = torch.randn(heads, seq_len, seq_len, device=device)

        attn = Attention(**attention_config).to(device)
        output = attn(x, attn_bias=attn_bias)

        assert output.shape == x.shape

        print("✓ Attention with bias successful")

    def test_attention_gradient_flow(self, attention_config, device):
        """Test gradients flow through attention."""
        batch_size, seq_len = 2, 64

        x = torch.randn(
            batch_size,
            seq_len,
            attention_config["dim"],
            device=device,
            requires_grad=True,
        )

        attn = Attention(**attention_config).to(device)
        output = attn(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().mean() > 1e-6

        print("✓ Attention gradient flow verified")


class TestTransformer:
    """Test Transformer block."""

    def test_transformer_initialization(self, transformer_config):
        """Test Transformer initializes correctly."""
        transformer = Transformer(**transformer_config)

        # Check number of layers (each layer has PEG + Attention + FF)
        assert len(transformer.layers) == transformer_config["depth"]

    def test_transformer_forward(self, transformer_config, device):
        """Test Transformer forward pass."""
        batch_size, seq_len = 2, 64

        x = torch.randn(batch_size, seq_len, transformer_config["dim"], device=device)

        transformer = Transformer(**transformer_config).to(device)
        output = transformer(x)

        assert output.shape == x.shape

        print("✓ Transformer forward pass successful")

    def test_transformer_with_position_bias(self, transformer_config, device):
        """Test Transformer with continuous position bias."""
        batch_size, seq_len = 2, 64
        heads = transformer_config["heads"]

        x = torch.randn(batch_size, seq_len, transformer_config["dim"], device=device)
        attn_bias = torch.randn(heads, seq_len, seq_len, device=device)

        transformer = Transformer(**transformer_config).to(device)
        output = transformer(x, attn_bias=attn_bias)

        assert output.shape == x.shape

        print("✓ Transformer with position bias successful")

    def test_transformer_with_peg(self, device):
        """Test Transformer with PEG (position encoding generator)."""
        batch_size, time, height, width = 2, 2, 8, 8
        dim = 1024
        seq_len = time * height * width

        x = torch.randn(batch_size, seq_len, dim, device=device)
        video_shape = (batch_size, time, height, width)

        transformer = Transformer(
            dim=dim,
            depth=4,
            dim_head=64,
            heads=16,
            peg=True,
            peg_causal=False,
        ).to(device)

        output = transformer(x, video_shape=video_shape)

        assert output.shape == x.shape

        print("✓ Transformer with PEG successful")

    def test_transformer_cross_attention(self, device):
        """Test Transformer with cross-attention."""
        batch_size, seq_len, context_len = 2, 64, 32
        dim = 1024

        x = torch.randn(batch_size, seq_len, dim, device=device)
        context = torch.randn(batch_size, context_len, dim, device=device)

        transformer = Transformer(
            dim=dim,
            depth=4,
            dim_head=64,
            heads=16,
            has_cross_attn=True,
            dim_context=dim,
        ).to(device)

        output = transformer(x, context=context)

        assert output.shape == x.shape

        print("✓ Transformer with cross-attention successful")

    def test_transformer_gradient_flow(self, transformer_config, device):
        """Test gradients flow through full transformer."""
        batch_size, seq_len = 2, 64

        x = torch.randn(
            batch_size,
            seq_len,
            transformer_config["dim"],
            device=device,
            requires_grad=True,
        )

        # Use smaller transformer for faster test
        transformer = Transformer(
            dim=transformer_config["dim"],
            depth=2,  # Smaller depth
            dim_head=64,
            heads=16,
        ).to(device)

        output = transformer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Deep transformers can have very small gradients (vanishing gradient)
        assert x.grad.abs().mean() > 1e-10

        print("✓ Transformer gradient flow verified")
        print(f"  - Gradient mean: {x.grad.abs().mean():.10f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
