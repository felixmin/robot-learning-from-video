"""
Tests for optical flow supervision module.

Tests FlowConfig validation, FlowDecoder architecture, and integration with LAM.
"""

import pytest
import torch

from lam.models.flow import (
    FlowConfig,
    FlowDecoder,
    RAFTTeacher,
    compute_flow_loss,
    compute_flow_summary_loss,
    compute_weighted_mean_flow,
)


class TestFlowConfig:
    """Test FlowConfig validation."""

    def test_valid_config_raft_small(self):
        """Test valid config with raft_small."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=0.1,
            decoder_depth=4,
        )
        assert config.model == "raft_small"
        assert config.loss_weight == 0.1
        assert config.decoder_depth == 4

    def test_valid_config_raft_large(self):
        """Test valid config with raft_large."""
        config = FlowConfig(
            model="raft_large",
            loss_weight=0.5,
            decoder_depth=8,
        )
        assert config.model == "raft_large"

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="must be 'raft_small' or 'raft_large'"):
            FlowConfig(
                model="invalid_model",
                loss_weight=0.1,
                decoder_depth=4,
            )

    def test_invalid_loss_weight_zero(self):
        """Test that zero loss weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=0.0,
                decoder_depth=4,
            )

    def test_invalid_loss_weight_negative(self):
        """Test that negative loss weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=-0.1,
                decoder_depth=4,
            )

    def test_invalid_decoder_depth(self):
        """Test that invalid decoder depth raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=0.1,
                decoder_depth=0,
            )

    def test_warmup_weight_at_zero(self):
        """Test that weight is 0 at step 0 with warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(0) == 0.0

    def test_warmup_weight_at_half(self):
        """Test that weight is 0.5 at halfway through warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(500) == 0.5

    def test_warmup_weight_at_end(self):
        """Test that weight is full after warmup completes."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=1.0,
            decoder_depth=4,
            warmup_steps=1000,
        )
        assert config.get_weight(1000) == 1.0
        assert config.get_weight(2000) == 1.0  # Stays at max

    def test_no_warmup(self):
        """Test that weight is full immediately with no warmup."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=0.5,
            decoder_depth=4,
            warmup_steps=0,
        )
        assert config.get_weight(0) == 0.5
        assert config.get_weight(100) == 0.5

    def test_summary_defaults_disabled(self):
        """Test summary loss defaults."""
        config = FlowConfig(
            model="raft_small",
            loss_weight=0.5,
            decoder_depth=4,
        )
        assert config.summary_loss_weight == 0.0
        assert config.summary_static_eps == 1e-6

    def test_invalid_summary_loss_weight_negative(self):
        """Test that negative summary loss weight raises error."""
        with pytest.raises(
            ValueError, match="summary_loss_weight must be non-negative"
        ):
            FlowConfig(
                model="raft_small",
                loss_weight=0.1,
                decoder_depth=4,
                summary_loss_weight=-0.01,
            )

    def test_invalid_summary_static_eps_non_positive(self):
        """Test that non-positive summary epsilon raises error."""
        with pytest.raises(ValueError, match="summary_static_eps must be positive"):
            FlowConfig(
                model="raft_small",
                loss_weight=0.1,
                decoder_depth=4,
                summary_static_eps=0.0,
            )


class TestFlowDecoder:
    """Test FlowDecoder architecture."""

    @pytest.fixture
    def decoder(self):
        """Create a small flow decoder for testing."""
        return FlowDecoder(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            image_size=(64, 64),
            effective_grid_size=(4, 4),
        )

    def test_output_shape(self, decoder):
        """Test that output has correct shape [B, 2, H, W]."""
        batch_size = 2
        h, w = 4, 4
        dim = 64

        context_tokens = torch.randn(batch_size, 1, h, w, dim)
        action_tokens = torch.randn(batch_size, 1, 1, 1, dim)  # code_seq_len=1
        attn_bias = torch.zeros(4, h * w, h * w)

        output = decoder(context_tokens, action_tokens, attn_bias)

        assert output.shape == (batch_size, 2, 64, 64)

    def test_gradient_flow(self, decoder):
        """Test that gradients flow through decoder."""
        context_tokens = torch.randn(2, 1, 4, 4, 64, requires_grad=True)
        action_tokens = torch.randn(2, 1, 1, 1, 64, requires_grad=True)
        attn_bias = torch.zeros(4, 16, 16)

        output = decoder(context_tokens, action_tokens, attn_bias)
        loss = output.sum()
        loss.backward()

        assert context_tokens.grad is not None
        assert action_tokens.grad is not None


class TestRAFTTeacher:
    """Test RAFTTeacher lazy loading and inference."""

    def test_lazy_loading(self):
        """Test that RAFT is not loaded until first use."""
        teacher = RAFTTeacher("raft_small")
        assert teacher._model is None

    def test_state_dict_empty(self):
        """Test that state_dict returns empty to avoid checkpoint pollution."""
        teacher = RAFTTeacher("raft_small")
        state_dict = teacher.state_dict()
        assert state_dict == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RAFT")
    def test_compute_flow_shape(self):
        """Test that compute_flow returns correct shape."""
        teacher = RAFTTeacher("raft_small")

        # RAFT requires minimum image size
        frame1 = torch.randn(1, 3, 1, 128, 128).cuda()
        frame2 = torch.randn(1, 3, 1, 128, 128).cuda()

        flow = teacher.compute_flow(frame1, frame2)

        assert flow.shape == (1, 2, 128, 128)
        assert flow.device.type == "cuda"


class TestComputeFlowLoss:
    """Test flow loss computation."""

    def test_mse_loss_normalized(self):
        """Test that flow loss is normalized MSE."""
        pred_flow = torch.randn(2, 2, 64, 64)
        gt_flow = torch.randn(2, 2, 64, 64)

        loss = compute_flow_loss(pred_flow, gt_flow, normalize=True)

        # Manually compute normalized MSE
        H, W = 64, 64
        norm = torch.tensor([W, H]).view(1, 2, 1, 1)
        pred_norm = pred_flow / norm
        gt_norm = gt_flow / norm
        expected = torch.nn.functional.mse_loss(pred_norm, gt_norm)
        assert torch.allclose(loss, expected)

    def test_mse_loss_unnormalized(self):
        """Test that unnormalized flow loss is raw MSE."""
        pred_flow = torch.randn(2, 2, 64, 64)
        gt_flow = torch.randn(2, 2, 64, 64)

        loss = compute_flow_loss(pred_flow, gt_flow, normalize=False)

        expected = torch.nn.functional.mse_loss(pred_flow, gt_flow)
        assert torch.allclose(loss, expected)

    def test_zero_loss_identical(self):
        """Test that identical flows give zero loss."""
        flow = torch.randn(2, 2, 64, 64)
        loss = compute_flow_loss(flow, flow)
        assert loss.item() == 0.0

    def test_normalization_reduces_scale(self):
        """Test that normalization brings flow to smaller scale."""
        # Large pixel displacements
        pred_flow = torch.randn(2, 2, 256, 256) * 100  # ~[-100, 100] pixels
        gt_flow = torch.randn(2, 2, 256, 256) * 100

        loss_unnorm = compute_flow_loss(pred_flow, gt_flow, normalize=False)
        loss_norm = compute_flow_loss(pred_flow, gt_flow, normalize=True)

        # Normalized loss should be much smaller (by factor of ~256^2)
        assert loss_norm < loss_unnorm / 1000


class TestFlowSummaryLoss:
    """Test weighted mean flow extraction and summary loss."""

    def test_weighted_mean_prefers_moving_pixels(self):
        flow = torch.zeros(1, 2, 2, 2)
        flow[:, 0] = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]])

        mean_dx, mean_dy = compute_weighted_mean_flow(flow)

        assert torch.allclose(mean_dx, torch.tensor([10.0]), atol=1e-5)
        assert torch.allclose(mean_dy, torch.tensor([0.0]), atol=1e-6)

    def test_weighted_mean_static_fallback_uses_plain_mean(self):
        flow = torch.zeros(1, 2, 2, 2)
        flow[:, 0] = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

        mean_dx, mean_dy = compute_weighted_mean_flow(flow, static_eps=1e6)

        assert torch.allclose(mean_dx, torch.tensor([4.0]), atol=1e-6)
        assert torch.allclose(mean_dy, torch.tensor([0.0]), atol=1e-6)

    def test_summary_loss_zero_for_identical_flows(self):
        flow = torch.randn(2, 2, 32, 32)
        loss = compute_flow_summary_loss(flow, flow)
        assert loss.item() == 0.0

    def test_summary_loss_penalizes_direction_mismatch(self):
        pred_flow = torch.zeros(1, 2, 16, 16)
        gt_flow = torch.zeros(1, 2, 16, 16)
        pred_flow[:, 0] = 1.0
        gt_flow[:, 0] = -1.0

        loss = compute_flow_summary_loss(pred_flow, gt_flow, normalize=False)
        assert loss.item() > 1.9

    def test_summary_normalization_reduces_scale(self):
        pred_flow = torch.ones(1, 2, 256, 256) * 10.0
        gt_flow = torch.zeros(1, 2, 256, 256)

        loss_unnorm = compute_flow_summary_loss(pred_flow, gt_flow, normalize=False)
        loss_norm = compute_flow_summary_loss(pred_flow, gt_flow, normalize=True)
        assert loss_norm < loss_unnorm / 1000


class TestLAMWithFlow:
    """Integration tests for LAM with flow supervision."""

    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "metrics": {
                    "log_every_n_steps": 1,
                    "num_unique_codes_every_n_steps": 1,
                },
                "optimizer": {
                    "type": "AdamW",
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
                "scheduler": {"type": "none"},
            }
        )

    @pytest.fixture
    def lam_config_with_flow(self):
        """Create minimal LAM config with flow for testing."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "dim": 64,
                "quant_dim": 16,
                "codebook_size": 8,
                "image_size": 128,
                "patch_size": 16,
                "spatial_depth": 2,
                "temporal_depth": 2,
                "dim_head": 16,
                "heads": 4,
                "code_seq_len": 1,
                "channels": 3,
                "attn_dropout": 0.0,
                "ff_dropout": 0.0,
                "vq_discarding_threshold": 0.1,
                "latent_ablation": "none",
                "use_dinov3_encoder": False,
                "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_pool_to_grid": None,
                "dino": {
                    "enabled": False,
                },
                "use_pixel_decoder": False,
                "use_aux_decoder": False,
                "codebook_replace_schedule": [[10, 100]],
                "flow": {
                    "enabled": True,
                    "model": "raft_small",
                    "loss_weight": 0.1,
                    "decoder_depth": 2,
                    "warmup_steps": 0,
                    "teacher_num_flow_updates": 4,
                    "teacher_chunk_size": 64,
                    "summary_loss_weight": 0.2,
                    "summary_static_eps": 1e-5,
                },
            }
        )

    @pytest.fixture
    def lam_config_without_flow(self):
        """Create minimal LAM config without flow for testing."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "dim": 64,
                "quant_dim": 16,
                "codebook_size": 8,
                "image_size": 128,
                "patch_size": 16,
                "spatial_depth": 2,
                "temporal_depth": 2,
                "dim_head": 16,
                "heads": 4,
                "code_seq_len": 1,
                "channels": 3,
                "attn_dropout": 0.0,
                "ff_dropout": 0.0,
                "vq_discarding_threshold": 0.1,
                "latent_ablation": "none",
                "use_dinov3_encoder": False,
                "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_pool_to_grid": None,
                "dino": {
                    "enabled": True,
                    "loss_weight": 1.0,
                    "warmup_steps": 0,
                },
                "use_pixel_decoder": False,
                "use_aux_decoder": False,
                "codebook_replace_schedule": [[10, 100]],
            }
        )

    @pytest.fixture
    def lam_config_with_flow_disabled(self):
        """Create minimal LAM config with flow explicitly disabled."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "dim": 64,
                "quant_dim": 16,
                "codebook_size": 8,
                "image_size": 128,
                "patch_size": 16,
                "spatial_depth": 2,
                "temporal_depth": 2,
                "dim_head": 16,
                "heads": 4,
                "code_seq_len": 1,
                "channels": 3,
                "attn_dropout": 0.0,
                "ff_dropout": 0.0,
                "vq_discarding_threshold": 0.1,
                "latent_ablation": "none",
                "use_dinov3_encoder": False,
                "dinov3_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_pool_to_grid": None,
                "dino": {
                    "enabled": True,
                    "loss_weight": 1.0,
                    "warmup_steps": 0,
                },
                "use_pixel_decoder": False,
                "use_aux_decoder": False,
                "codebook_replace_schedule": [[10, 100]],
                "flow": {
                    "enabled": False,
                },
            }
        )

    def test_flow_config_parsed(self, lam_config_with_flow, training_config):
        """Test that flow config is correctly parsed in task."""
        from lam.task import LAMTask

        task = LAMTask(
            model_config=lam_config_with_flow,
            training_config=training_config,
        )

        assert task.model.flow_config is not None
        assert task.model.flow_config.model == "raft_small"
        assert task.model.flow_config.summary_loss_weight == 0.2
        assert task.model.flow_config.summary_static_eps == pytest.approx(1e-5)
        assert task.model.flow_decoder is not None
        assert task.model.flow_teacher is not None

    def test_no_flow_when_not_configured(
        self, lam_config_without_flow, training_config
    ):
        """Test that missing flow config fails fast."""
        from lam.task import LAMTask
        from omegaconf.errors import ConfigAttributeError

        with pytest.raises(ConfigAttributeError):
            LAMTask(
                model_config=lam_config_without_flow,
                training_config=training_config,
            )

    def test_no_flow_when_explicitly_disabled(
        self, lam_config_with_flow_disabled, training_config
    ):
        """Test that flow is disabled when configured with enabled=false."""
        from lam.task import LAMTask

        task = LAMTask(
            model_config=lam_config_with_flow_disabled,
            training_config=training_config,
        )

        assert task.model.flow_config is None
        assert task.model.flow_decoder is None
        assert task.model.flow_teacher is None


class TestHydraConfigWithFlow:
    """Test Hydra configuration with flow supervision."""

    @pytest.fixture
    def config_dir(self):
        """Get path to config directory."""
        from pathlib import Path

        return str(Path(__file__).parent.parent / "config")

    def test_stage1_local_loads(self, config_dir):
        """Test that stage1_local config loads correctly."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_local"])

            # Validate experiment
            assert cfg.experiment.name == "stage1_local"

            # Validate flow config
            assert cfg.model.flow.model == "raft_large"
            assert cfg.model.flow.loss_weight == 10.0
            assert cfg.model.flow.decoder_depth == 4
            assert cfg.model.flow.warmup_steps == 100000

            # Validate decoder flags/config
            assert cfg.model.dino.enabled is True
            assert cfg.model.dino.warmup_steps == 0
            assert isinstance(cfg.model.use_aux_decoder, bool)

            # Validate flow visualization strategy
            assert cfg.validation.strategies.flow_visualization.enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
