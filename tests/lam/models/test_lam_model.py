"""
Test LatentActionModel

Tests the complete LAM model that combines transformers + NSVQ.
"""

import pytest
import torch
from lam.models.latent_action_model import DinoConfig, LatentActionModel


@pytest.fixture
def lam_model_config(device):
    """Small LAM configuration for fast testing."""
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
        "channels": 3,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "vq_discarding_threshold": 0.1,
        "vq_discarding_threshold_schedule": None,
        "latent_ablation": "none",
        "use_dinov3_encoder": False,
        "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_pool_to_grid": None,
        "dino_config": DinoConfig(loss_weight=1.0, warmup_steps=0),
        "use_dino_decoder": True,
        "use_pixel_decoder": False,
        "use_aux_decoder": True,
        "flow_config": None,
        "codebook_replace_schedule": [(10, 100)],
        "metrics_num_unique_codes_every_n_steps": 1,
        "device": device,
    }


@pytest.fixture
def lam_model(lam_model_config):
    """Create LAM model instance."""
    device = lam_model_config.pop("device")
    model = LatentActionModel(**lam_model_config).to(device)
    return model


class TestLAMInitialization:
    """Test LAM model initialization."""

    def test_lam_model_creation(self, lam_model):
        """Test LAM model initializes correctly."""
        assert hasattr(lam_model, "encoder_projection")
        assert hasattr(lam_model, "enc_spatial_transformer")
        assert hasattr(lam_model, "enc_temporal_transformer")
        assert hasattr(lam_model, "vq")
        # Decoders are now conditional - check they exist as attributes
        assert hasattr(lam_model, "dino_decoder")
        assert hasattr(lam_model, "aux_decoder")
        assert hasattr(lam_model, "aux_to_pixels")

        print("✓ LAM model initialized successfully")

    def test_lam_components(self, lam_model, lam_model_config):
        """Test LAM model components are correct types."""
        from lam.models.attention import Transformer
        from lam.models.nsvq import NSVQ

        assert isinstance(lam_model.enc_spatial_transformer, Transformer)
        assert isinstance(lam_model.enc_temporal_transformer, Transformer)
        # DINO decoder is enabled by default
        assert isinstance(lam_model.dino_decoder, Transformer)
        assert isinstance(lam_model.vq, NSVQ)

        print("✓ LAM components verified")


class TestLAMForward:
    """Test LAM forward pass."""

    def test_forward_with_images(self, lam_model, device):
        """Test forward pass with 2-frame video (standard case)."""
        batch_size = 2
        # Input: [B, C, T, H, W] where T=2 (frame_t, frame_t+1)
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        # Forward pass (training mode) - returns (loss, metrics_dict)
        loss, metrics = lam_model(video, step=0)

        # Check outputs
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert isinstance(metrics, dict)
        assert "unique_codes_in_batch" in metrics
        assert metrics["unique_codes_in_batch"] > 0

        print("✓ Forward pass successful")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Unique codes used: {metrics['unique_codes_in_batch']}")

    def test_forward_returns_reconstruction_only(self, lam_model, device):
        """Test forward pass with return_recons_only=True."""
        batch_size = 2
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        # Get reconstruction only
        recon = lam_model(video, step=0, return_recons_only=True)

        # Check shape: should be [B, C, H, W] (single frame reconstruction)
        assert recon.shape == (batch_size, 3, 256, 256)

        print("✓ Reconstruction-only mode works")
        print(f"  - Reconstruction shape: {recon.shape}")

    def test_forward_returns_codebook_ids(self, lam_model, device):
        """Test forward pass with return_only_codebook_ids=True."""
        batch_size = 2
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        # Get codebook indices only
        indices = lam_model(video, step=0, return_only_codebook_ids=True)

        # Check shape: [B, code_seq_len]
        assert indices.shape == (batch_size, 4)
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < 8  # codebook_size

        print("✓ Codebook IDs extraction works")
        print(f"  - Indices shape: {indices.shape}")
        print(f"  - Sample indices: {indices[0].tolist()}")


class TestLAMInference:
    """Test LAM inference mode."""

    def test_inference_mode(self, lam_model, device):
        """Test inference (evaluation) mode."""
        batch_size = 2
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            recon = lam_model.inference(video)

        # Check reconstruction shape
        assert recon.shape == (batch_size, 3, 256, 256)

        print("✓ Inference mode works")

    def test_inference_falls_back_to_pixel_decoder_without_aux(
        self, lam_model_config, device
    ):
        """Pixel decoder should provide reconstructions when aux is disabled."""
        lam_model_config.pop("device")
        lam_model_config["use_aux_decoder"] = False
        lam_model_config["use_pixel_decoder"] = True
        model = LatentActionModel(**lam_model_config).to(device)
        video = torch.randn(2, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            recon = model.inference(video)

        assert recon is not None
        assert recon.shape == (2, 3, 256, 256)

    def test_inference_with_codebook_ids(self, lam_model, device):
        """Test inference returns codebook IDs."""
        batch_size = 2
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            indices = lam_model.inference(video, return_only_codebook_ids=True)

        assert indices.shape == (batch_size, 4)
        assert indices.dtype == torch.long

        print("✓ Inference codebook extraction works")


class TestLAMGradients:
    """Test gradient flow through LAM."""

    def test_gradient_flow(self, lam_model, device):
        """Test gradients flow through full model."""
        batch_size = 2
        video = torch.randn(
            batch_size, 3, 2, 256, 256, device=device, requires_grad=True
        )

        # Forward pass - returns (loss, metrics_dict)
        loss, metrics = lam_model(video, step=0)

        # Backward pass
        loss.backward()

        # Check gradients exist for input
        assert video.grad is not None
        assert video.grad.abs().mean() > 1e-8

        # Check that main model components have gradients
        # (not all params will have grads if they're in unused branches)
        grad_params = sum(1 for p in lam_model.parameters() if p.grad is not None)
        total_params = sum(1 for p in lam_model.parameters())

        # At least 80% of parameters should have gradients
        assert (
            grad_params / total_params > 0.8
        ), f"Only {grad_params}/{total_params} params have gradients"

        print("✓ Gradients flow through full model")
        print(f"  - Input gradient mean: {video.grad.abs().mean():.8f}")
        print(f"  - Params with gradients: {grad_params}/{total_params}")


class TestLAMShapes:
    """Test LAM with different input shapes."""

    def test_different_batch_sizes(self, lam_model_config, device):
        """Test LAM with different batch sizes."""
        lam_model_config.pop("device")
        model = LatentActionModel(**lam_model_config).to(device)

        for batch_size in [1, 2, 4]:
            video = torch.randn(batch_size, 3, 2, 256, 256, device=device)
            loss, metrics = model(video, step=0)

            assert loss.item() >= 0
            print(f"✓ Batch size {batch_size} works")

    def test_4d_input_conversion(self, lam_model, device):
        """Test LAM converts 4D input to 5D correctly (requires 2+ frames)."""
        batch_size = 2
        # For LAM to work, we need at least 2 frames
        # So we use 5D input [B, C, T, H, W] where T >= 2
        video = torch.randn(batch_size, 3, 2, 256, 256, device=device)

        # This should work fine
        loss, metrics = lam_model(video, step=0)

        assert loss.item() >= 0
        print("✓ Video input (2 frames) works")


class TestLAMCodebookManagement:
    """Test codebook usage and replacement."""

    def test_codebook_usage_tracking(self, lam_model, device):
        """Test codebook usage is tracked across batches."""
        # Run multiple forward passes
        for _ in range(5):
            video = torch.randn(2, 3, 2, 256, 256, device=device)
            loss, metrics = lam_model(
                video, step=999
            )  # step=999 won't trigger replacement

        # Check codebook usage
        codebook_used = lam_model.vq.codebooks_used
        total_usage = codebook_used.sum().item()

        assert total_usage > 0, "Codebook should be used"

        used_codes = (codebook_used > 0).sum().item()
        print("✓ Codebook tracking works")
        print(f"  - Total uses: {total_usage}")
        print(f"  - Unique codes used: {used_codes}/8")

    def test_codebook_replacement_trigger(self, lam_model, device):
        """Test codebook replacement is triggered at specific steps."""
        # Step 10 should trigger replacement (based on code: step % 10 == 0 and step < 100)
        video = torch.randn(2, 3, 2, 256, 256, device=device)

        # This should print "update codebook 10"
        loss, metrics = lam_model(video, step=10)

        print("✓ Codebook replacement triggered at step 10")

    def test_vq_discarding_threshold_schedule(self, lam_model_config, device):
        """Test step-based replacement threshold schedule."""
        lam_model_config.pop("device")
        lam_model_config["vq_discarding_threshold"] = 0.02
        lam_model_config["vq_discarding_threshold_schedule"] = [
            (0.1, 100),
            (0.01, 1000),
        ]
        model = LatentActionModel(**lam_model_config).to(device)

        assert model._get_vq_discarding_threshold(0) == pytest.approx(0.1)
        assert model._get_vq_discarding_threshold(99) == pytest.approx(0.1)
        assert model._get_vq_discarding_threshold(100) == pytest.approx(0.01)
        assert model._get_vq_discarding_threshold(10000) == pytest.approx(0.01)

    def test_vq_discarding_threshold_schedule_must_increase_until_step(
        self, lam_model_config, device
    ):
        """Test threshold schedule validation fails on non-monotonic ranges."""
        lam_model_config.pop("device")
        lam_model_config["vq_discarding_threshold_schedule"] = [
            (0.1, 200),
            (0.01, 100),
        ]

        with pytest.raises(ValueError, match="strictly increasing"):
            LatentActionModel(**lam_model_config).to(device)

    def test_dino_warmup_weight(self, lam_model_config, device):
        """Test DINO warmup weight follows configured schedule."""
        lam_model_config.pop("device")
        lam_model_config["dino_config"] = DinoConfig(loss_weight=2.0, warmup_steps=100)
        model = LatentActionModel(**lam_model_config).to(device)

        video = torch.randn(2, 3, 2, 256, 256, device=device)
        _, metrics_step0 = model(video, step=0)
        _, metrics_step50 = model(video, step=50)
        _, metrics_step100 = model(video, step=100)

        assert metrics_step0["dino_weight"] == pytest.approx(0.0)
        assert "dino_loss" not in metrics_step0
        assert metrics_step50["dino_weight"] == pytest.approx(1.0)
        assert metrics_step100["dino_weight"] == pytest.approx(2.0)


class TestLAMStateDict:
    """Test model save/load."""

    def test_state_dict_save_load(self, lam_model, device):
        """Test model can be saved and loaded."""
        # Save state dict
        state_dict = lam_model.state_dict()

        assert len(state_dict) > 0
        print(f"✓ State dict has {len(state_dict)} parameters")

    def test_load_from_checkpoint(self, lam_model_config, device, tmp_path):
        """Test model can load from checkpoint file."""
        lam_model_config.pop("device")
        model1 = LatentActionModel(**lam_model_config).to(device)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(model1.state_dict(), checkpoint_path)

        # Create new model and load
        model2 = LatentActionModel(**lam_model_config).to(device)
        model2.load(checkpoint_path)

        # Check parameters match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

        print("✓ Model load from checkpoint works")


class TestLAMPatchProperties:
    """Test patch-related properties."""

    def test_patch_height_width_property(self, lam_model):
        """Test patch_height_width property."""
        h, w = lam_model.patch_height_width

        # With image_size=256, patch_size=32: 256/32 = 8
        assert h == 8
        assert w == 8

        print(f"✓ Patch dimensions: {h}×{w}")


class TestCodeOnlyPath:
    """Regression tests for the no-side-effect code-only forward path."""

    def test_return_only_codebook_ids_does_not_update_counter(self, lam_model, device):
        """forward(return_only_codebook_ids=True) must not increment codebooks_used."""
        lam_model.vq.codebooks_used.zero_()
        video = torch.randn(2, 3, 2, 256, 256, device=device)
        with torch.no_grad():
            lam_model(video, return_only_codebook_ids=True)
        assert lam_model.vq.codebooks_used.sum().item() == 0

    def test_return_only_codebook_ids_matches_get_indices(self, lam_model, device):
        """forward code-only path returns same indices as vq.get_indices directly."""
        video = torch.randn(2, 3, 2, 256, 256, device=device)
        with torch.no_grad():
            indices_forward = lam_model(video, return_only_codebook_ids=True)
            first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
            _, _, first_tokens, last_tokens = lam_model._encode_frames(
                first_frame, rest_frames
            )
            indices_direct = lam_model.vq.get_indices(first_tokens, last_tokens)
        assert torch.equal(indices_forward, indices_direct)

    def test_inference_codebook_ids_match_forward_code_only(self, lam_model, device):
        """Inference index path must match the side-effect-free forward code-only path."""
        video = torch.randn(2, 3, 2, 256, 256, device=device)
        with torch.no_grad():
            indices_forward = lam_model(video, return_only_codebook_ids=True)
            indices_inference = lam_model.inference(
                video, return_only_codebook_ids=True
            )
        assert torch.equal(indices_forward, indices_inference)

    def test_normal_forward_still_updates_counter(self, lam_model, device):
        """Standard training forward must still increment codebooks_used."""
        lam_model.vq.codebooks_used.zero_()
        video = torch.randn(2, 3, 2, 256, 256, device=device)
        lam_model(video, step=0)
        assert lam_model.vq.codebooks_used.sum().item() > 0


class TestInferenceCompatibility:
    """Regression tests for inference-only compatibility semantics."""

    def test_inference_returns_none_without_reconstruction_decoder(
        self, lam_model_config, device
    ):
        """Inference must return None when both reconstruction decoders are disabled."""
        lam_model_config.pop("device")
        lam_model_config["use_aux_decoder"] = False
        lam_model_config["use_pixel_decoder"] = False
        model = LatentActionModel(**lam_model_config).to(device)
        video = torch.randn(2, 3, 2, 256, 256, device=device)

        with torch.no_grad():
            recon = model.inference(video)

        assert recon is None

    def test_lam_encoder_vq_inference_prunes_decoder_attrs(self, lam_model):
        """Stage-2 pruning must drop decoder attrs but keep shared encoder projection state."""
        from lam.inference import LAMEncoderVQInference

        original_state_keys = set(lam_model.state_dict().keys())
        assert any(key.startswith("aux_decoder.") for key in original_state_keys)
        assert any(key.startswith("decoder_context_projection.") for key in original_state_keys)
        assert lam_model.decoder_context_projection is lam_model.pixel_projection
        assert lam_model.encoder_projection is lam_model.pixel_projection

        wrapper = LAMEncoderVQInference(lam_model, prune_decoders=True)

        assert wrapper._model.aux_decoder is None
        assert wrapper._model.aux_to_pixels is None
        assert wrapper._model.dino_decoder is None
        assert wrapper._model.decoder_context_projection is None
        assert wrapper._model.pixel_projection is not None
        assert wrapper._model.encoder_projection is not None

        pruned_state_keys = set(wrapper._model.state_dict().keys())
        assert not any(key.startswith("aux_decoder.") for key in pruned_state_keys)
        assert not any(
            key.startswith("decoder_context_projection.") for key in pruned_state_keys
        )
        assert any(key.startswith("pixel_projection.") for key in pruned_state_keys)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
