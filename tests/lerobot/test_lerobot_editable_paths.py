from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
from tests.helpers.paths import script_path


def _load_script_module(script_name: str, module_name: str):
    path = script_path(script_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_editable_paths_accept_legacy_single_path() -> None:
    mod = _load_script_module("6_train_lerobot.py", "stage3_train_script")
    cfg = OmegaConf.create({"lerobot": {"install_editable": "lerobot_policy_hlrp"}})

    editable_paths = mod._editable_paths_from_cfg(cfg)

    assert editable_paths == [
        Path(mod.workspace_root / "lerobot_policy_hlrp").resolve()
    ]


def test_train_editable_paths_preserve_order_and_dedupe() -> None:
    mod = _load_script_module("6_train_lerobot.py", "stage3_train_script")
    cfg = OmegaConf.create(
        {
            "lerobot": {
                "install_editables": ["lerobot", "lerobot_policy_hlrp", "lerobot"],
            }
        }
    )

    editable_paths = mod._editable_paths_from_cfg(cfg)

    assert editable_paths == [
        Path(mod.workspace_root / "lerobot").resolve(),
        Path(mod.workspace_root / "lerobot_policy_hlrp").resolve(),
    ]


def test_rollout_editable_paths_accept_legacy_single_path() -> None:
    mod = _load_script_module("7_rollout_lerobot.py", "stage3_rollout_script")
    cfg = OmegaConf.create(
        {"lerobot_eval": {"install_policy_editable": "lerobot_policy_hlrp"}}
    )

    editable_paths = mod._editable_paths_from_cfg(cfg)

    assert editable_paths == [
        Path(mod.workspace_root / "lerobot_policy_hlrp").resolve()
    ]


def test_rollout_editable_paths_preserve_order_and_dedupe() -> None:
    mod = _load_script_module("7_rollout_lerobot.py", "stage3_rollout_script")
    cfg = OmegaConf.create(
        {
            "lerobot_eval": {
                "install_editables": ["lerobot", "lerobot_policy_hlrp", "lerobot"],
            }
        }
    )

    editable_paths = mod._editable_paths_from_cfg(cfg)

    assert editable_paths == [
        Path(mod.workspace_root / "lerobot").resolve(),
        Path(mod.workspace_root / "lerobot_policy_hlrp").resolve(),
    ]


def test_rollout_command_inherits_saved_train_env_overrides(tmp_path) -> None:
    mod = _load_script_module("7_rollout_lerobot.py", "stage3_rollout_script")
    policy_dir = tmp_path / "pretrained_model"
    policy_dir.mkdir()
    with open(policy_dir / "train_config.json", "w") as f:
        json.dump(
            {
                "env": {
                    "observation_height": 512,
                    "observation_width": 512,
                    "camera_name": "agentview_image,robot0_eye_in_hand_image",
                    "control_mode": "relative",
                }
            },
            f,
        )

    cfg = OmegaConf.create(
        {
            "lerobot_eval": {
                "command": "lerobot-eval",
                "policy_path": str(policy_dir),
                "env_type": "libero",
                "env_task": "libero_10",
                "eval_n_episodes": 10,
                "eval_batch_size": 1,
                "extra_args": [],
            }
        }
    )

    cmd = mod._command_from_cfg(cfg)

    assert "--env.observation_height=512" in cmd
    assert "--env.observation_width=512" in cmd
    assert "--env.camera_name=agentview_image,robot0_eye_in_hand_image" in cmd
    assert "--env.control_mode=relative" in cmd


def test_rollout_command_prefers_explicit_env_extra_args_over_inherited(
    tmp_path,
) -> None:
    mod = _load_script_module("7_rollout_lerobot.py", "stage3_rollout_script")
    policy_dir = tmp_path / "pretrained_model"
    policy_dir.mkdir()
    with open(policy_dir / "train_config.json", "w") as f:
        json.dump({"env": {"observation_height": 512}}, f)

    cfg = OmegaConf.create(
        {
            "lerobot_eval": {
                "command": "lerobot-eval",
                "policy_path": str(policy_dir),
                "env_type": "libero",
                "env_task": "libero_10",
                "eval_n_episodes": 10,
                "eval_batch_size": 1,
                "extra_args": ["--env.observation_height=360"],
            }
        }
    )

    cmd = mod._command_from_cfg(cfg)

    assert "--env.observation_height=360" in cmd
    assert "--env.observation_height=512" not in cmd
