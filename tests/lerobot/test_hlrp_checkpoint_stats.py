from __future__ import annotations

import json

import pytest

from lerobot_policy_hlrp.policies.hlrp_smolvla_shared import checkpoint_stats


def test_require_saved_normalization_stats_raises_for_scratch_when_missing(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="missing hlrp_normalization_stats.json"):
        checkpoint_stats.require_saved_normalization_stats(
            tmp_path,
            init_mode="scratch",
        )


def test_load_saved_normalization_stats_reads_local_sidecar(tmp_path) -> None:
    stats = {
        "action": {"mean": [0.0], "std": [1.0]},
        "observation.state": {"mean": [1.0], "std": [2.0]},
    }
    path = tmp_path / checkpoint_stats.NORMALIZATION_STATS_FILENAME
    with open(path, "w") as f:
        json.dump(stats, f)

    loaded = checkpoint_stats.load_saved_normalization_stats(tmp_path)

    assert loaded == stats


def test_write_normalization_stats_from_train_config_rebuilds_sidecar(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_dir = tmp_path / "pretrained_model"
    policy_dir.mkdir()
    train_config_path = policy_dir / checkpoint_stats.TRAIN_CONFIG_FILENAME
    with open(train_config_path, "w") as f:
        json.dump(
            {
                "dataset": {
                    "repo_id": "HuggingFaceVLA/libero",
                    "root": str(tmp_path / "dataset_root"),
                    "revision": "v3.0",
                }
            },
            f,
        )

    expected_stats = {
        "action": {"mean": [0.0], "std": [1.0]},
        "observation.state": {"mean": [1.0], "std": [2.0]},
    }

    class _FakeDatasetMetadata:
        def __init__(self, repo_id: str, root: str | None = None, revision: str | None = None):
            assert repo_id == "HuggingFaceVLA/libero"
            assert root == str(tmp_path / "dataset_root")
            assert revision == "v3.0"
            self.stats = expected_stats

    monkeypatch.setattr(
        checkpoint_stats,
        "LeRobotDatasetMetadata",
        _FakeDatasetMetadata,
    )

    out_path = checkpoint_stats.write_normalization_stats_from_train_config(policy_dir)

    assert out_path == policy_dir / checkpoint_stats.NORMALIZATION_STATS_FILENAME
    with open(out_path) as f:
        repaired = json.load(f)
    assert repaired == expected_stats
