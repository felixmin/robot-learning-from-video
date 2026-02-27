from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_policy_hlrp" / "src"))
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
            stage2_artifact=None,
        )


def test_scratch_mode_requires_null_stage2_artifact() -> None:
    with pytest.raises(ValueError, match="requires stage2_artifact=null"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="action",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            stage2_artifact=Path("/tmp/art.pt"),
        )


def test_artifact_mode_with_artifact_is_valid() -> None:
    cfg = HLRPSmolVLASharedConfig(
        init_mode="artifact",
        stage3_training_mode="action",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
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
        stage2_artifact=None,
    )
    assert cfg.init_mode == "scratch"
    assert cfg.stage2_artifact is None


def test_latent_mode_requires_laq_fields() -> None:
    with pytest.raises(ValueError, match="laq_checkpoint_path"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="latent",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            stage2_artifact=None,
        )


def test_alternating_mode_requires_schedule() -> None:
    with pytest.raises(ValueError, match="alternating_latent_steps"):
        HLRPSmolVLASharedConfig(
            init_mode="scratch",
            stage3_training_mode="alternating",
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
            stage2_artifact=None,
            laq_checkpoint_path=Path("/tmp/laq.ckpt"),
            laq_future_frames=10,
            laq_camera_keys=("observation.images.image",),
            alternating_supervised_mode="action",
        )
