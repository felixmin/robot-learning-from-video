"""
Test LAQ Lightning task wrapper.

Tests LAQTask with mock configs and synthetic data.
"""

import pytest
import torch
from omegaconf import OmegaConf

from laq.task import LAQTask, separate_weight_decayable_params


@pytest.fixture
def model_config():
    """Minimal model config for testing."""
    return OmegaConf.create({
        "dim": 256,
        "quant_dim": 16,
        "codebook_size": 8,
        "image_size": 256,  # Match existing tests
        "patch_size": 32,
        "spatial_depth": 2,
        "temporal_depth": 2,
        "dim_head": 32,
        "heads": 4,
        "code_seq_len": 4,
        "channels": 3,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
    })


@pytest.fixture
def training_config():
    """Minimal training config for testing."""
    return OmegaConf.create({
        "max_steps": 10,
        "metrics": {
            "log_every_n_steps": 1,
            "num_unique_codes_every_n_steps": 1,
        },
        "optimizer": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
        "scheduler": {
            "type": "cosine",
            "min_lr": 1e-6,
        },
        "gradient": {
            "clip_val": 1.0,
        },
    })


@pytest.fixture
def synthetic_batch():
    """Synthetic frame pair batch [B, C, 2, H, W]."""
    return torch.randn(2, 3, 2, 256, 256)


class TestSeparateWeightDecayableParams:
    """Test weight decay parameter separation."""

    def test_separate_2d_and_less(self):
        """Test separating 2D+ params from <2D params."""
        # Create dummy parameters
        weight_2d = torch.nn.Parameter(torch.randn(10, 10))  # 2D -> weight decay
        bias_1d = torch.nn.Parameter(torch.randn(10))        # 1D -> no weight decay
        weight_3d = torch.nn.Parameter(torch.randn(5, 5, 5)) # 3D -> weight decay

        params = [weight_2d, bias_1d, weight_3d]
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        assert len(wd_params) == 2
        assert len(no_wd_params) == 1
        # Use 'is' for identity check, not 'in' (which does element-wise comparison)
        assert any(p is weight_2d for p in wd_params)
        assert any(p is weight_3d for p in wd_params)
        assert any(p is bias_1d for p in no_wd_params)

        print("✓ Weight decay parameter separation works")

    def test_all_2d_params(self):
        """Test with all 2D+ params."""
        params = [
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5)),
        ]
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        assert len(wd_params) == 2
        assert len(no_wd_params) == 0

    def test_all_1d_params(self):
        """Test with all <2D params."""
        params = [
            torch.nn.Parameter(torch.randn(10)),
            torch.nn.Parameter(torch.randn(5)),
        ]
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        assert len(wd_params) == 0
        assert len(no_wd_params) == 2


class TestLAQTaskInitialization:
    """Test LAQTask initialization."""

    def test_task_creation(self, model_config, training_config):
        """Test LAQTask initializes correctly."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
            use_ema=False,
        )

        assert task.model is not None
        assert task.model_config == model_config
        assert task.training_config == training_config
        assert task.use_ema is False

        print("✓ LAQTask initialized")

    def test_task_with_ema(self, model_config, training_config):
        """Test LAQTask with EMA enabled."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
            use_ema=True,
        )

        assert task.use_ema is True

        print("✓ LAQTask with EMA initialized")

    def test_task_passes_vq_discarding_threshold_schedule(self, model_config, training_config):
        """Test task forwards threshold schedule to LAQ model."""
        model_config.vq_discarding_threshold_schedule = [[0.1, 100], [0.01, 1000]]
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
            use_ema=False,
        )

        assert task.model.vq_discarding_threshold_schedule == [(0.1, 100), (0.01, 1000)]

    def test_task_builds_dino_config(self, model_config, training_config):
        """Test task parses dino block into model dino config."""
        model_config.dino = {"enabled": True, "loss_weight": 2.0, "warmup_steps": 123}
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
            use_ema=False,
        )

        assert task.model.dino_config is not None
        assert task.model.dino_config.loss_weight == 2.0
        assert task.model.dino_config.warmup_steps == 123

    def test_task_disables_dino_when_configured(self, model_config, training_config):
        """Test dino decoder is disabled with dino.enabled=false."""
        model_config.dino = {"enabled": False}
        model_config.use_pixel_decoder = True
        model_config.flow = None
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
            use_ema=False,
        )

        assert task.model.dino_config is None
        assert task.model.dino_decoder is None


class TestLAQTaskForward:
    """Test LAQTask forward pass."""

    def test_forward_pass(self, model_config, training_config, synthetic_batch, device):
        """Test forward pass through task."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Forward pass - returns (loss, metrics_dict)
        loss, metrics = task(batch, step=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
        assert isinstance(metrics, dict)
        assert metrics["unique_codes_in_batch"] > 0
        assert metrics["entries_at_or_above_usage_threshold_count_in_window"] > 0
        assert metrics["usage_count_threshold_in_window"] >= 0
        assert float(metrics["code_assignments_in_window"].item()) > 0

        print(f"✓ Forward pass: loss={loss.item():.4f}, num_unique={metrics['unique_codes_in_batch']}")

    def test_forward_with_recons_only(self, model_config, training_config, synthetic_batch, device):
        """Test forward pass returning reconstructions only."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Forward pass with recons only
        recons = task(batch, step=0, return_recons_only=True)

        assert recons.shape == (2, 3, 256, 256)  # [B, C, H, W]

        print(f"✓ Forward pass (recons only): shape={recons.shape}")


class TestLAQTaskTrainingStep:
    """Test LAQTask training step."""

    def test_training_step_tensor(self, model_config, training_config, synthetic_batch, device):
        """Test training step with tensor batch."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Training step
        loss = task.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        print(f"✓ Training step (tensor): loss={loss.item():.4f}")

    def test_training_step_dict(self, model_config, training_config, synthetic_batch, device):
        """Test training step with metadata dict batch."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        # Create metadata dict batch
        batch_dict = {
            "frames": synthetic_batch.to(device),
            "scene_idx": [0, 1],
        }

        # Training step
        loss = task.training_step(batch_dict, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

        print(f"✓ Training step (dict): loss={loss.item():.4f}")


class TestLAQTaskValidationStep:
    """Test LAQTask validation step."""

    def test_validation_step(self, model_config, training_config, synthetic_batch, device):
        """Test validation step."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Validation step
        loss = task.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

        # Check that validation batch was stored
        assert task.validation_batch is not None
        assert task.validation_batch.shape[0] <= 8  # Max 8 samples

        print(f"✓ Validation step: loss={loss.item():.4f}")

    def test_validation_batch_storage(self, model_config, training_config, device):
        """Test validation batch storage."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        # Create large batch (> 8 samples)
        large_batch = torch.randn(16, 3, 2, 256, 256).to(device)

        # Validation step
        task.validation_step(large_batch, batch_idx=0)

        # Should store max 8 samples
        assert task.validation_batch.shape[0] == 8

        print("✓ Validation batch storage (max 8 samples)")

    def test_validation_epoch_end(self, model_config, training_config, synthetic_batch, device):
        """Test validation epoch end."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Store validation batch
        task.validation_step(batch, batch_idx=0)
        assert task.validation_batch is not None

        # End of epoch should reset
        task.on_validation_epoch_end()
        assert task.validation_batch is None

        print("✓ Validation epoch end resets batch")


class TestLAQTaskOptimizer:
    """Test LAQTask optimizer configuration."""

    def test_configure_optimizers(self, model_config, training_config):
        """Test optimizer configuration."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        )

        opt_config = task.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

        optimizer = opt_config["optimizer"]
        assert len(optimizer.param_groups) == 2  # Two groups (wd / no_wd)

        # Check weight decay settings
        assert optimizer.param_groups[0]["weight_decay"] == 0.01  # wd group
        assert optimizer.param_groups[1]["weight_decay"] == 0.0   # no_wd group

        print("✓ Optimizer configured with 2 parameter groups")

    def test_lr_scheduler(self, model_config, training_config):
        """Test LR scheduler configuration."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        )

        opt_config = task.configure_optimizers()
        scheduler_config = opt_config["lr_scheduler"]

        assert scheduler_config["interval"] == "step"
        assert scheduler_config["frequency"] == 1

        print("✓ LR scheduler configured")


class TestLAQTaskReconstructions:
    """Test LAQTask reconstruction generation."""

    def test_generate_reconstructions(self, model_config, training_config, synthetic_batch, device):
        """Test generating reconstructions for visualization."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Generate reconstructions
        recons = task.generate_reconstructions(batch)

        assert recons.shape == (2, 3, 256, 256)  # [B, C, H, W]
        assert recons.device == batch.device

        print(f"✓ Generated reconstructions: shape={recons.shape}")


class TestLAQTaskGradients:
    """Test LAQTask gradient flow."""

    def test_gradient_flow(self, model_config, training_config, synthetic_batch, device):
        """Test gradients flow through task."""
        task = LAQTask(
            model_config=model_config,
            training_config=training_config,
        ).to(device)

        batch = synthetic_batch.to(device)

        # Forward + backward
        loss = task.training_step(batch, batch_idx=0)
        loss.backward()

        # Check gradients exist
        params_with_grad = [
            p for p in task.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]

        total_params = sum(1 for _ in task.parameters())
        grad_ratio = len(params_with_grad) / total_params

        assert grad_ratio > 0.5  # At least 50% of params have gradients

        print(f"✓ Gradient flow: {len(params_with_grad)}/{total_params} params with gradients")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
