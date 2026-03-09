from __future__ import annotations

import importlib.util
from pathlib import Path

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
