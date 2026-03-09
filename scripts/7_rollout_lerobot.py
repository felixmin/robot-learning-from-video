#!/usr/bin/env python3
# ruff: noqa: E402
"""
Script 7: Stage-3 LeRobot rollout/evaluation entrypoint.

This script is intended to run inside the Slurm container launched by
`scripts/submit_job.py`. It can optionally editable-install mounted LeRobot
packages and then execute `lerobot-eval`.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.libero_runtime import ensure_libero_config
from common.run_context import setup_run_context


def _to_bool_flag(value: object) -> str:
    return "true" if bool(value) else "false"


def _run_install_command(
    cmd: list[str],
    *,
    logger,
    cwd: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    logger.info("Install command: %s", shlex.join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        err = f"Command failed ({e.returncode}): {shlex.join(cmd)}"
        logger.warning(err)
        return False, err


def _install_editable_package(
    *,
    editable_path: Path,
    logger,
    cwd: Path,
    env: dict[str, str],
) -> None:
    python = sys.executable
    attempted_errors: list[str] = []

    ok, err = _run_install_command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(editable_path),
        ],
        logger=logger,
        cwd=cwd,
        env=env,
    )
    if ok:
        return
    attempted_errors.append(err)

    if "No module named pip" in err:
        logger.info("pip module missing; attempting bootstrap via ensurepip")
        boot_ok, boot_err = _run_install_command(
            [python, "-m", "ensurepip", "--upgrade"],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if boot_ok:
            ok, err = _run_install_command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(editable_path),
                ],
                logger=logger,
                cwd=cwd,
                env=env,
            )
            if ok:
                return
            attempted_errors.append(err)
        else:
            attempted_errors.append(boot_err)

    uv_bin = shutil.which("uv", path=env.get("PATH"))
    if uv_bin is not None:
        ok, err = _run_install_command(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(editable_path),
            ],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if ok:
            return
        attempted_errors.append(err)
    else:
        attempted_errors.append("uv executable not found on PATH")

    pip_bin = shutil.which("pip", path=env.get("PATH"))
    if pip_bin is not None:
        ok, err = _run_install_command(
            [
                pip_bin,
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(editable_path),
            ],
            logger=logger,
            cwd=cwd,
            env=env,
        )
        if ok:
            return
        attempted_errors.append(err)
    else:
        attempted_errors.append("pip executable not found on PATH")

    details = "\n".join(f"- {msg}" for msg in attempted_errors)
    raise RuntimeError(
        "Failed to editable-install package with all supported installers:\n"
        f"{details}"
    )


def _editable_paths_from_cfg(cfg: DictConfig) -> list[Path]:
    raw_paths = OmegaConf.select(cfg, "lerobot_eval.install_editables")
    if raw_paths is None:
        legacy_path = OmegaConf.select(cfg, "lerobot_eval.install_policy_editable")
        raw_paths = [] if legacy_path is None else [legacy_path]
    elif not (OmegaConf.is_list(raw_paths) or isinstance(raw_paths, (list, tuple))):
        raise ValueError(
            "lerobot_eval.install_editables must be a list of package paths"
        )

    editable_paths: list[Path] = []
    seen_paths: set[str] = set()
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        editable_path = Path(str(raw_path))
        if not editable_path.is_absolute():
            editable_path = workspace_root / editable_path
        editable_path = editable_path.resolve()
        if not editable_path.exists():
            raise FileNotFoundError(f"Editable package path not found: {editable_path}")
        editable_key = str(editable_path)
        if editable_key in seen_paths:
            continue
        seen_paths.add(editable_key)
        editable_paths.append(editable_path)

    return editable_paths


def _runtime_cwd_from_cfg(cfg: DictConfig) -> Path:
    run_dir = OmegaConf.select(cfg, "logging.runs_dir")
    if run_dir is None:
        raise ValueError("logging.runs_dir must be set for stage-3 runs")
    cwd = Path(str(run_dir))
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _command_from_cfg(cfg: DictConfig) -> list[str]:
    eval_cmd = str(OmegaConf.select(cfg, "lerobot_eval.command") or "lerobot-eval")

    policy_path = OmegaConf.select(cfg, "lerobot_eval.policy_path")
    env_type = OmegaConf.select(cfg, "lerobot_eval.env_type")
    n_episodes = OmegaConf.select(cfg, "lerobot_eval.eval_n_episodes")
    batch_size = OmegaConf.select(cfg, "lerobot_eval.eval_batch_size")

    required = {
        "lerobot_eval.policy_path": policy_path,
        "lerobot_eval.env_type": env_type,
        "lerobot_eval.eval_n_episodes": n_episodes,
        "lerobot_eval.eval_batch_size": batch_size,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required lerobot_eval config keys: {missing}")

    cmd = [
        eval_cmd,
        f"--policy.path={policy_path}",
        f"--env.type={env_type}",
        f"--eval.n_episodes={int(n_episodes)}",
        f"--eval.batch_size={int(batch_size)}",
    ]

    env_task = OmegaConf.select(cfg, "lerobot_eval.env_task")
    if env_task:
        cmd.append(f"--env.task={env_task}")

    policy_device = OmegaConf.select(cfg, "lerobot_eval.policy_device")
    if policy_device is not None:
        cmd.append(f"--policy.device={policy_device}")

    policy_use_amp = OmegaConf.select(cfg, "lerobot_eval.policy_use_amp")
    if policy_use_amp is not None:
        cmd.append(f"--policy.use_amp={_to_bool_flag(policy_use_amp)}")

    output_dir = OmegaConf.select(cfg, "lerobot_eval.output_dir")
    if output_dir:
        cmd.append(f"--output_dir={output_dir}")

    job_name = OmegaConf.select(cfg, "lerobot_eval.job_name")
    if job_name:
        cmd.append(f"--job_name={job_name}")

    seed = OmegaConf.select(cfg, "lerobot_eval.seed")
    if seed is not None:
        cmd.append(f"--seed={int(seed)}")

    trust_remote_code = OmegaConf.select(cfg, "lerobot_eval.trust_remote_code")
    if trust_remote_code is not None:
        cmd.append(f"--trust_remote_code={_to_bool_flag(trust_remote_code)}")

    extra_args = OmegaConf.select(cfg, "lerobot_eval.extra_args") or []
    if not (OmegaConf.is_list(extra_args) or isinstance(extra_args, (list, tuple))):
        raise ValueError("lerobot_eval.extra_args must be a list of strings")
    stripped_extra_args: list[str] = []
    for i, arg in enumerate(extra_args):
        if not isinstance(arg, str):
            raise ValueError(f"lerobot_eval.extra_args[{i}] must be a string")
        if arg.strip():
            stripped_extra_args.append(arg.strip())

    cmd.extend(
        _inherit_train_env_args(
            policy_path=Path(str(policy_path)),
            explicit_args=cmd + stripped_extra_args,
        )
    )
    cmd.extend(stripped_extra_args)

    return cmd


def _inherit_train_env_args(
    *,
    policy_path: Path,
    explicit_args: list[str],
) -> list[str]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.is_file():
        return []

    with open(train_config_path) as f:
        train_cfg = json.load(f)
    env_cfg = train_cfg.get("env")
    if not isinstance(env_cfg, dict):
        return []

    inherited_keys = (
        "obs_type",
        "render_mode",
        "camera_name",
        "camera_name_mapping",
        "init_states",
        "observation_height",
        "observation_width",
        "control_mode",
        "episode_length",
    )
    explicit_env_keys = {
        arg.split("=", 1)[0][len("--env.") :]
        for arg in explicit_args
        if arg.startswith("--env.") and "=" in arg
    }

    inherited_args: list[str] = []
    for key in inherited_keys:
        if key in explicit_env_keys:
            continue
        value = env_cfg.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        inherited_args.append(f"--env.{key}={rendered}")
    return inherited_args


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, _ = setup_run_context(
        cfg=cfg,
        workspace_root=workspace_root,
        logger_name="lerobot.rollout",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 3: LeRobot Rollout")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    env = os.environ.copy()
    packages_path = str(workspace_root / "packages")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        packages_path
        if not existing_pythonpath
        else f"{packages_path}:{existing_pythonpath}"
    )
    env_overrides = OmegaConf.select(cfg, "lerobot_eval.env") or {}
    if not OmegaConf.is_dict(env_overrides):
        raise ValueError("lerobot_eval.env must be a mapping of environment variables")
    for k, v in env_overrides.items():
        if v is None:
            continue
        env[str(k)] = str(v)
    ensure_libero_config(env=env, logger=logger)

    for editable_path in _editable_paths_from_cfg(cfg):
        logger.info("Installing editable package: %s", editable_path)
        _install_editable_package(
            editable_path=editable_path,
            logger=logger,
            cwd=workspace_root,
            env=env,
        )

    cmd = _command_from_cfg(cfg)
    runtime_cwd = _runtime_cwd_from_cfg(cfg)
    logger.info("Launching LeRobot rollout command:")
    logger.info("  %s", shlex.join(cmd))
    logger.info("  cwd=%s", runtime_cwd)
    subprocess.run(cmd, cwd=str(runtime_cwd), env=env, check=True)
    logger.info("Stage 3 rollout complete.")


if __name__ == "__main__":
    main()
