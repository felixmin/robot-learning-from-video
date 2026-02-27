#!/usr/bin/env python3
"""
Script 6: Stage-3 LeRobot fine-tuning/evaluation entrypoint.

This script is intended to run inside the Slurm container launched by
`scripts/submit_job.py`. It can optionally editable-install a local LeRobot
policy plugin and then execute `lerobot-train`.
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
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.cache_env import configure_cache_env, resolve_cache_dir
from common.unified_logging import resolve_runs_dir, setup_unified_logging


def _to_bool_flag(value: object) -> str:
    return "true" if bool(value) else "false"


def _episodes_arg(value: object) -> str | None:
    if value is None:
        return None
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    s = str(value).strip()
    return s if s else None


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


def _install_editable_policy(
    *,
    editable_path: Path,
    logger,
    cwd: Path,
    env: dict[str, str],
) -> None:
    python = sys.executable
    attempted_errors: list[str] = []

    # 1) Prefer the current interpreter's pip module.
    ok, err = _run_install_command(
        [python, "-m", "pip", "install", "--no-deps", "-e", str(editable_path)],
        logger=logger,
        cwd=cwd,
        env=env,
    )
    if ok:
        return
    attempted_errors.append(err)

    # 1b) If pip module is missing, try bootstrapping it.
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
                [python, "-m", "pip", "install", "--no-deps", "-e", str(editable_path)],
                logger=logger,
                cwd=cwd,
                env=env,
            )
            if ok:
                return
            attempted_errors.append(err)
        else:
            attempted_errors.append(boot_err)

    # 2) Prefer uv pip with explicit interpreter (installs into the active env).
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

    # 3) Last resort: pip executable on PATH (may target user/system site).
    pip_bin = shutil.which("pip", path=env.get("PATH"))
    if pip_bin is not None:
        ok, err = _run_install_command(
            [pip_bin, "install", "--no-deps", "-e", str(editable_path)],
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
        "Failed to editable-install policy package with all supported installers:\n"
        f"{details}"
    )


def _lerobot_run_command_from_cfg(cfg: DictConfig) -> list[str]:
    train_cmd_raw = OmegaConf.select(cfg, "lerobot.command")

    policy_type = OmegaConf.select(cfg, "lerobot.policy_type")
    policy_repo_id = OmegaConf.select(cfg, "lerobot.policy_repo_id")
    init_mode = OmegaConf.select(cfg, "lerobot.init_mode")
    dataset_repo_id = OmegaConf.select(cfg, "lerobot.dataset_repo_id")
    output_dir = OmegaConf.select(cfg, "lerobot.output_dir")
    job_name = OmegaConf.select(cfg, "lerobot.job_name")
    steps = OmegaConf.select(cfg, "lerobot.steps")
    batch_size = OmegaConf.select(cfg, "lerobot.batch_size")
    num_workers = OmegaConf.select(cfg, "lerobot.num_workers")
    eval_freq = OmegaConf.select(cfg, "lerobot.eval_freq")
    eval_batch_size = OmegaConf.select(cfg, "lerobot.eval_batch_size")
    log_freq = OmegaConf.select(cfg, "lerobot.log_freq")
    save_freq = OmegaConf.select(cfg, "lerobot.save_freq")
    stage2_artifact = OmegaConf.select(cfg, "lerobot.stage2_artifact")

    required = {
        "lerobot.command": train_cmd_raw,
        "lerobot.policy_type": policy_type,
        "lerobot.policy_repo_id": policy_repo_id,
        "lerobot.dataset_repo_id": dataset_repo_id,
        "lerobot.output_dir": output_dir,
        "lerobot.job_name": job_name,
        "lerobot.steps": steps,
        "lerobot.batch_size": batch_size,
        "lerobot.num_workers": num_workers,
        "lerobot.eval_freq": eval_freq,
        "lerobot.log_freq": log_freq,
        "lerobot.save_freq": save_freq,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required lerobot config keys: {missing}")
    if not isinstance(train_cmd_raw, str) or not train_cmd_raw.strip():
        raise ValueError("lerobot.command must be a non-empty string")
    train_cmd = train_cmd_raw

    cmd = [
        train_cmd,
        f"--policy.type={policy_type}",
        f"--policy.repo_id={policy_repo_id}",
        f"--policy.push_to_hub={_to_bool_flag(OmegaConf.select(cfg, 'lerobot.push_to_hub') is True)}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={int(steps)}",
        f"--batch_size={int(batch_size)}",
        f"--num_workers={int(num_workers)}",
        f"--eval_freq={int(eval_freq)}",
        f"--log_freq={int(log_freq)}",
        f"--save_freq={int(save_freq)}",
        f"--wandb.enable={_to_bool_flag(OmegaConf.select(cfg, 'lerobot.wandb_enable') is True)}",
    ]

    if str(policy_type) == "hlrp_smolvla_shared":
        if init_mode is None:
            raise ValueError("lerobot.init_mode is required for policy_type=hlrp_smolvla_shared")
        cmd.append(f"--policy.init_mode={init_mode}")
        if str(init_mode) == "artifact":
            if stage2_artifact is None:
                raise ValueError("lerobot.stage2_artifact is required when lerobot.init_mode=artifact")
            cmd.append(f"--policy.stage2_artifact={stage2_artifact}")
        elif str(init_mode) == "scratch":
            if stage2_artifact is not None:
                raise ValueError("lerobot.stage2_artifact must be null when lerobot.init_mode=scratch")
            cmd.append("--policy.stage2_artifact=null")
        else:
            raise ValueError(f"Unsupported lerobot.init_mode={init_mode!r}; expected 'artifact' or 'scratch'")

    if eval_batch_size is not None:
        cmd.append(f"--eval.batch_size={int(eval_batch_size)}")

    policy_device = OmegaConf.select(cfg, "lerobot.policy_device")
    if policy_device is not None:
        cmd.append(f"--policy.device={policy_device}")

    episodes = _episodes_arg(OmegaConf.select(cfg, "lerobot.dataset_episodes"))
    if episodes is not None:
        cmd.append(f"--dataset.episodes={episodes}")

    env_type = OmegaConf.select(cfg, "lerobot.env_type")
    if env_type:
        cmd.append(f"--env.type={env_type}")
    env_task = OmegaConf.select(cfg, "lerobot.env_task")
    if env_task:
        cmd.append(f"--env.task={env_task}")

    extra_args = OmegaConf.select(cfg, "lerobot.extra_args") or []
    if not (OmegaConf.is_list(extra_args) or isinstance(extra_args, (list, tuple))):
        raise ValueError("lerobot.extra_args must be a list of strings")
    for i, arg in enumerate(extra_args):
        if not isinstance(arg, str):
            raise ValueError(f"lerobot.extra_args[{i}] must be a string")
        if arg.strip():
            cmd.append(arg.strip())
    return cmd


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None
    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=OmegaConf.select(cfg, "logging.root_dir"),
            logging_runs_dir=OmegaConf.select(cfg, "logging.runs_dir"),
            workspace_root=workspace_root,
            experiment_name=OmegaConf.select(cfg, "experiment.name"),
        )

    logger, _ = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=OmegaConf.select(cfg, "logging.job_id"),
        log_level=str(OmegaConf.select(cfg, "logging.level") or "INFO"),
        logger_name="lerobot.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 3: LeRobot Training")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    env = os.environ.copy()
    packages_path = str(workspace_root / "packages")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        packages_path if not existing_pythonpath else f"{packages_path}:{existing_pythonpath}"
    )
    env_overrides = OmegaConf.select(cfg, "lerobot.env") or {}
    if not OmegaConf.is_dict(env_overrides):
        raise ValueError("lerobot.env must be a mapping of environment variables")
    for k, v in env_overrides.items():
        if v is None:
            continue
        env[str(k)] = str(v)

    install_editable = OmegaConf.select(cfg, "lerobot.install_policy_editable")
    if install_editable:
        editable_path = Path(str(install_editable))
        if not editable_path.is_absolute():
            editable_path = workspace_root / editable_path
        if not editable_path.exists():
            raise FileNotFoundError(f"Editable policy path not found: {editable_path}")
        logger.info("Installing editable policy package: %s", editable_path)
        _install_editable_policy(
            editable_path=editable_path,
            logger=logger,
            cwd=workspace_root,
            env=env,
        )

    cmd = _lerobot_run_command_from_cfg(cfg)
    logger.info("Launching LeRobot command:")
    logger.info("  %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(workspace_root), env=env, check=True)
    logger.info("Stage 3 training complete.")


if __name__ == "__main__":
    main()
