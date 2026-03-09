from __future__ import annotations

from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.configuration_hlrp_smolvla_shared import (
    HLRPSmolVLASharedConfig,
)
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.modeling_hlrp_smolvla_shared import (
    HLRPSmolVLASharedPolicy,
    NORMALIZATION_STATS_FILENAME,
)


def _make_config() -> HLRPSmolVLASharedConfig:
    return HLRPSmolVLASharedConfig(
        init_mode="scratch",
        stage3_training_mode="action",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        action_subset_ratio=1.0,
        action_subset_key="episode_index",
        latent_scope="all",
        stage2_artifact=None,
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(512, 512, 3),
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )


def test_save_pretrained_persists_normalization_stats(tmp_path: Path) -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = _make_config()
    policy.core = torch.nn.Linear(1, 1)
    policy.normalization_stats = {
        "action": {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([3.0, 4.0])},
        "observation.state": {"mean": [0.5], "std": [2.5]},
    }

    policy._save_pretrained(tmp_path)

    stats_path = tmp_path / NORMALIZATION_STATS_FILENAME
    assert stats_path.is_file()
    loaded = HLRPSmolVLASharedPolicy._load_saved_normalization_stats(tmp_path)
    assert loaded == {
        "action": {"mean": [1.0, 2.0], "std": [3.0, 4.0]},
        "observation.state": {"mean": [0.5], "std": [2.5]},
    }
