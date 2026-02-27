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
        HLRPSmolVLASharedConfig(init_mode="artifact", stage2_artifact=None)


def test_scratch_mode_requires_null_stage2_artifact() -> None:
    with pytest.raises(ValueError, match="requires stage2_artifact=null"):
        HLRPSmolVLASharedConfig(init_mode="scratch", stage2_artifact=Path("/tmp/art.pt"))


def test_artifact_mode_with_artifact_is_valid() -> None:
    cfg = HLRPSmolVLASharedConfig(init_mode="artifact", stage2_artifact=Path("/tmp/art.pt"))
    assert cfg.init_mode == "artifact"
    assert cfg.stage2_artifact == Path("/tmp/art.pt")


def test_scratch_mode_with_null_artifact_is_valid() -> None:
    cfg = HLRPSmolVLASharedConfig(init_mode="scratch", stage2_artifact=None)
    assert cfg.init_mode == "scratch"
    assert cfg.stage2_artifact is None
