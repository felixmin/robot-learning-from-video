from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.configuration_hlrp_smolvla_shared import (
    HLRPSmolVLASharedConfig,
)


def test_artifact_mode_requires_stage2_artifact() -> None:
    with pytest.raises(ValueError, match="requires non-null stage2_artifact"):
        HLRPSmolVLASharedConfig(
            init_mode="artifact",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="index",
            latent_scope="all",
            stage2_artifact=None,
        )


def test_scratch_mode_requires_null_stage2_artifact() -> None:
    with pytest.raises(ValueError, match="requires stage2_artifact=null"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="index",
            latent_scope="all",
            stage2_artifact=Path("/tmp/art.pt"),
        )


def test_artifact_mode_with_artifact_is_valid() -> None:
    cfg = HLRPSmolVLASharedConfig(
        init_mode="artifact",
        stage3_training_mode="action",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        action_subset_ratio=1.0,
        action_subset_key="index",
        latent_scope="all",
        stage2_artifact=Path("/tmp/art.pt"),
    )
    assert cfg.init_mode == "artifact"
    assert cfg.stage2_artifact == Path("/tmp/art.pt")


def test_scratch_mode_with_null_artifact_is_valid() -> None:
    cfg = HLRPSmolVLASharedConfig(
        init_mode="scratch",
        stage3_training_mode="action",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        action_subset_ratio=1.0,
        action_subset_key="index",
        latent_scope="all",
        stage2_artifact=None,
    )
    assert cfg.init_mode == "scratch"
    assert cfg.stage2_artifact is None


def test_latent_mode_requires_lam_fields() -> None:
    with pytest.raises(ValueError, match="lam_checkpoint_path"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="latent",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="index",
            latent_scope="all",
            stage2_artifact=None,
        )


def test_alternating_mode_requires_schedule() -> None:
    with pytest.raises(ValueError, match="alternating_latent_steps_per_action_step"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="alternating",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="index",
            latent_scope="all",
            stage2_artifact=None,
            lam_checkpoint_path=Path("/tmp/lam.ckpt"),
            lam_future_frames=10,
            lam_camera_keys=("observation.images.image",),
        )


def test_action_subset_ratio_must_be_valid() -> None:
    with pytest.raises(ValueError, match="action_subset_ratio"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=0.0,
            action_subset_key="index",
            latent_scope="all",
            stage2_artifact=None,
        )


def test_action_subset_key_must_be_valid() -> None:
    with pytest.raises(ValueError, match="action_subset_key"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="task_index",
            latent_scope="all",
            stage2_artifact=None,
        )


def test_latent_scope_must_be_valid() -> None:
    with pytest.raises(ValueError, match="latent_scope"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            action_subset_ratio=1.0,
            action_subset_key="index",
            latent_scope="subset",
            stage2_artifact=None,
        )
