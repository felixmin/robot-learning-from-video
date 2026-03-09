from __future__ import annotations

import importlib.util

import pytest
from hydra import compose, initialize_config_dir
from tests.helpers.paths import CONFIG_DIR, script_path


def _load_stage3_script_module():
    path = script_path("6_train_lerobot.py")
    spec = importlib.util.spec_from_file_location("stage3_train_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_dir() -> str:
    return str(CONFIG_DIR)


def test_command_builder_artifact_mode_includes_stage2_artifact(
    config_dir: str,
) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "lerobot.policy.init_mode=artifact",
                "lerobot.policy.stage2_artifact=/tmp/stage2.pt",
            ],
        )
    cmd = mod._lerobot_run_command_from_cfg(cfg)
    assert "--policy.init_mode=artifact" in cmd
    assert "--policy.stage2_artifact=/tmp/stage2.pt" in cmd


def test_command_builder_scratch_mode_sets_null_stage2_artifact(
    config_dir: str,
) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=stage3_local"],
        )
    cmd = mod._lerobot_run_command_from_cfg(cfg)
    assert "--policy.init_mode=scratch" in cmd
    assert "--policy.stage2_artifact=null" in cmd


def test_command_builder_artifact_mode_requires_stage2_artifact(
    config_dir: str,
) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "lerobot.policy.init_mode=artifact",
                "lerobot.policy.stage2_artifact=null",
            ],
        )
    with pytest.raises(
        ValueError, match="required when lerobot.policy.init_mode=artifact"
    ):
        mod._lerobot_run_command_from_cfg(cfg)


def test_command_builder_requires_non_null_command(config_dir: str) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "lerobot.command=null",
            ],
        )
    with pytest.raises(ValueError, match="Missing required lerobot config keys"):
        mod._lerobot_run_command_from_cfg(cfg)


def test_command_builder_forwards_grad_accum_steps(config_dir: str) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "stage3_profile=multitask_scratch",
                "lerobot.grad_accum_steps=2",
            ],
        )
    cmd = mod._lerobot_run_command_from_cfg(cfg)
    assert "--grad_accum_steps=2" in cmd


def test_command_builder_forwards_stage3_dataset_episode_subset(config_dir: str) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "stage3_dataset=libero_5pct",
            ],
        )
    cmd = mod._lerobot_run_command_from_cfg(cfg)
    assert (
        "--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83]"
        in cmd
    )


def test_command_builder_forwards_random_stage3_dataset_episode_subset(
    config_dir: str,
) -> None:
    mod = _load_stage3_script_module()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=stage3_local",
                "stage3_dataset=libero_5pct_random",
            ],
        )
    cmd = mod._lerobot_run_command_from_cfg(cfg)
    assert (
        "--dataset.episodes=[13,51,54,61,65,88,93,142,161,163,178,189,191,198,206,209,228,255,285,318,326,333,393,407,429,440,447,451,457,466,476,501,541,563,569,592,600,689,696,704,727,735,740,747,758,775,778,859,864,865,919,928,940,1034,1098,1116,1130,1149,1182,1206,1209,1232,1236,1266,1287,1301,1309,1330,1354,1385,1429,1436,1442,1466,1494,1508,1516,1518,1554,1563,1583,1650,1652,1657]"
        in cmd
    )
