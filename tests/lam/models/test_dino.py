"""
Test DINOv3 Integration

Tests the DINOFeatureExtractor, DINOEncoder, and DINOWrapper modules,
as well as their integration with LAM.
"""

import os
from pathlib import Path
import warnings

import pytest
import torch
import torch.nn as nn


# Skip all tests if transformers not available
pytest.importorskip("transformers")


def _has_hf_auth_token() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return True
    return (Path.home() / ".huggingface" / "token").exists()


if not _has_hf_auth_token():
    message = (
        "DINO tests require Hugging Face auth for the gated model "
        "`facebook/dinov3-vits16-pretrain-lvd1689m`. "
        "Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`, or create `~/.huggingface/token`. "
        "Skipping the DINO test module."
    )
    warnings.warn(message)
    pytest.skip(message, allow_module_level=True)


@pytest.fixture
def dino_model_name():
    """Default DINO model for testing."""
    return "facebook/dinov3-vits16-pretrain-lvd1689m"


class TestDINOFeatureExtractor:
    """Test DINOFeatureExtractor module."""

    def test_feature_extractor_initialization(self, device, dino_model_name):
        """Test DINOFeatureExtractor initializes correctly."""
        from lam.models.dino import DINOFeatureExtractor

        extractor = DINOFeatureExtractor(
            model_name=dino_model_name, freeze=True, target_size=256
        )

        assert extractor.patch_size == 16  # DINOv3 uses 16x16 patches
        assert extractor.target_size == 256  # 256 is divisible by 16
        assert extractor.hidden_size == 384  # ViT-S has 384 dim

        # Check frozen
        for param in extractor.model.parameters():
            assert not param.requires_grad

        print("✓ DINOFeatureExtractor initialized correctly")

    def test_feature_extractor_output_shape(self, device, dino_model_name):
        """Test feature extractor output shape."""
        from lam.models.dino import DINOFeatureExtractor

        extractor = DINOFeatureExtractor(
            model_name=dino_model_name, target_size=256
        ).to(device)

        # Input: [B, C, H, W]
        x = torch.randn(2, 3, 256, 256, device=device)

        with torch.no_grad():
            features = extractor(x)

        # Output: [B, H_grid, W_grid, D]
        # 256 / 16 = 16 patches per side
        assert features.shape == (2, 16, 16, 384)

        print(f"✓ Feature extractor output shape: {features.shape}")

    def test_feature_extractor_multi_layer(self, device, dino_model_name):
        """Test multi-layer feature extraction."""
        from lam.models.dino import DINOFeatureExtractor

        extractor = DINOFeatureExtractor(
            model_name=dino_model_name, target_size=256
        ).to(device)

        x = torch.randn(2, 3, 256, 256, device=device)
        layers = [6, 11]

        with torch.no_grad():
            features = extractor(x, output_hidden_states=True, layer_indices=layers)

        assert len(features) == 2
        for feat in features:
            assert feat.shape == (2, 16, 16, 384)

        print(f"✓ Multi-layer extraction works: {len(features)} layers")

    def test_feature_extractor_size_adjustment(self, device, dino_model_name):
        """Test automatic size adjustment to patch multiple."""
        from lam.models.dino import DINOFeatureExtractor

        # 250 is not divisible by 16, should adjust to 256 (16*16)
        extractor = DINOFeatureExtractor(model_name=dino_model_name, target_size=250)

        # Should round to nearest multiple of 16
        assert extractor.target_size == 256  # 16 * 16 = 256

        print(f"✓ Size adjustment: 250 -> {extractor.target_size}")


class TestDINOEncoder:
    """Test DINOEncoder module."""

    def test_encoder_initialization(self, device, dino_model_name):
        """Test DINOEncoder initializes correctly."""
        from lam.models.dino import DINOFeatureExtractor, DINOEncoder

        extractor = DINOFeatureExtractor(model_name=dino_model_name, target_size=256)
        encoder = DINOEncoder(extractor, out_dim=512).to(device)

        assert encoder.in_dim == 384
        assert encoder.out_dim == 512
        assert isinstance(encoder.proj, nn.Linear)
        # assert isinstance(encoder.norm, nn.LayerNorm)

        print("✓ DINOEncoder initialized correctly")

    def test_encoder_output_shape(self, device, dino_model_name):
        """Test DINOEncoder output shape."""
        from lam.models.dino import DINOFeatureExtractor, DINOEncoder

        extractor = DINOFeatureExtractor(model_name=dino_model_name, target_size=256)
        encoder = DINOEncoder(extractor, out_dim=512).to(device)

        x = torch.randn(2, 3, 256, 256, device=device)

        with torch.no_grad():
            features = encoder(x)

        # Output: [B, 1, H_grid, W_grid, D_out]
        assert features.shape == (2, 1, 16, 16, 512)

        print(f"✓ DINOEncoder output shape: {features.shape}")


class TestDINOWrapper:
    """Test DINOWrapper module."""

    def test_wrapper_squeeze(self, device, dino_model_name):
        """Test DINOWrapper correctly squeezes time dimension."""
        from lam.models.dino import DINOFeatureExtractor, DINOEncoder, DINOWrapper

        extractor = DINOFeatureExtractor(model_name=dino_model_name, target_size=256)
        encoder = DINOEncoder(extractor, out_dim=512)
        wrapper = DINOWrapper(encoder).to(device)

        # Input: [B, C, 1, H, W] (with time dim)
        x = torch.randn(2, 3, 1, 256, 256, device=device)

        with torch.no_grad():
            features = wrapper(x)

        # Output: [B, 1, H_grid, W_grid, D]
        assert features.shape == (2, 1, 16, 16, 512)

        print(f"✓ DINOWrapper output shape: {features.shape}")


class TestLAMWithDINO:
    """Test LAM model with DINO encoder."""

    @pytest.fixture
    def lam_with_dino(self, device, dino_model_name):
        """Create LAM model with DINO encoder."""
        from lam.models.latent_action_quantization import (
            LatentActionQuantization,
            DinoConfig,
        )

        model = LatentActionQuantization(
            dim=512,
            quant_dim=32,
            codebook_size=8,
            image_size=256,
            patch_size=32,  # Will be overridden by DINO
            spatial_depth=2,
            temporal_depth=2,
            heads=4,
            dim_head=32,
            code_seq_len=4,
            channels=3,
            attn_dropout=0.0,
            ff_dropout=0.0,
            vq_discarding_threshold=0.1,
            vq_discarding_threshold_schedule=None,
            latent_ablation="none",
            metrics_num_unique_codes_every_n_steps=1,
            dino_config=DinoConfig(loss_weight=1.0, warmup_steps=0),
            use_dino_decoder=True,
            use_pixel_decoder=False,
            use_aux_decoder=True,
            flow_config=None,
            codebook_replace_schedule=[(10, 100)],
            use_dinov3_encoder=True,
            dinov3_model_name=dino_model_name,
            dinov3_pool_to_grid=None,
        ).to(device)
        return model

    def test_lam_dino_grid_size(self, lam_with_dino):
        """Test LAM with DINO has correct grid size."""
        h, w = lam_with_dino.patch_height_width

        # DINO patch_size=16, image=256 -> 16x16 grid
        assert h == 16
        assert w == 16

        print(f"✓ LAM+DINO grid size: {h}×{w}")

    def test_lam_dino_forward(self, lam_with_dino, device):
        """Test LAM+DINO forward pass."""
        x = torch.randn(2, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            loss, metrics = lam_with_dino(x, step=0)

        assert loss.item() >= 0
        assert metrics["unique_codes_in_batch"] > 0
        assert "dino_loss" in metrics
        assert "aux_loss" in metrics

        print(
            f"✓ LAM+DINO forward: loss={loss.item():.4f}, unique={metrics['unique_codes_in_batch']}"
        )

    def test_lam_dino_reconstruction(self, lam_with_dino, device):
        """Test LAM+DINO reconstruction output."""
        x = torch.randn(2, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            recon = lam_with_dino(x, step=0, return_recons_only=True)

        assert recon.shape == (2, 3, 256, 256)

        print(f"✓ LAM+DINO reconstruction shape: {recon.shape}")

    def test_lam_dino_codebook_ids(self, lam_with_dino, device):
        """Test LAM+DINO codebook ID extraction."""
        x = torch.randn(2, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            indices = lam_with_dino(x, step=0, return_only_codebook_ids=True)

        assert indices.shape == (2, 4)  # code_seq_len=4
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < 8  # codebook_size

        print(f"✓ LAM+DINO codebook indices: {indices[0].tolist()}")

    def test_lam_dino_gradient_flow(self, lam_with_dino, device):
        """Test gradients flow through LAM+DINO (DINO frozen)."""
        x = torch.randn(2, 3, 2, 256, 256, device=device, requires_grad=True)

        loss, metrics = lam_with_dino(x, step=0)
        loss.backward()

        # Input should have gradients (through unfrozen parts)
        assert x.grad is not None

        # DINO extractor should NOT have gradients (frozen)
        dino_params = list(lam_with_dino.dino_feature_extractor.model.parameters())
        for p in dino_params[:5]:  # Check first 5 params
            assert p.grad is None or p.grad.abs().sum() == 0

        # Projection layer SHOULD have gradients
        assert lam_with_dino.dino_encoder.proj.weight.grad is not None

        print("✓ Gradient flow correct (DINO frozen, projection trainable)")


class TestNSVQWithDINOGrid:
    """Test NSVQ adapts to DINO's 16x16 grid."""

    def test_nsvq_dynamic_cnn(self, device):
        """Test NSVQ builds correct CNN for 16x16 grid."""
        from lam.models.nsvq import NSVQ

        # With 16x16 grid and code_seq_len=4, need to go to 2x2
        vq = NSVQ(
            dim=512,
            num_embeddings=8,
            embedding_dim=32,
            device=device,
            code_seq_len=4,
            grid_size=(16, 16),
        ).to(device)

        assert vq.grid_h == 16
        assert vq.grid_w == 16

        # NSVQ expects flattened input: [B, h*w, dim]
        first_tokens = torch.randn(2, 256, 512, device=device)  # [B, 16*16, 512]
        last_tokens = torch.randn(2, 256, 512, device=device)

        with torch.no_grad():
            out, perp, usage, indices = vq(first_tokens, last_tokens)

        # Output shape: [B, code_seq_len, dim] for the quantized representation
        assert out.shape == (2, 4, 512)  # code_seq_len=4
        assert indices.shape == (2, 4)

        print(
            f"✓ NSVQ with 16x16 grid works, output shape: {out.shape}, indices shape: {indices.shape}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
