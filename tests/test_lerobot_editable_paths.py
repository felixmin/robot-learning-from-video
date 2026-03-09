from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def _load_script_module(script_name: str, module_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {script_path}")
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
