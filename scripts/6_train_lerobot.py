#!/usr/bin/env python3
"""
Script 6: Stage-3 LeRobot fine-tuning/evaluation entrypoint.

This script is intended to run inside the Slurm container launched by
`scripts/submit_job.py`. It can optionally editable-install mounted LeRobot
packages and then execute `lerobot-train`.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.run_context import setup_run_context


def _to_bool_flag(value: object) -> str:
    return "true" if bool(value) else "false"


def _episodes_arg(value: object) -> str | None:
    if value is None:
        return None
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    s = str(value).strip()
    return s if s else None


def _format_cli_value(value: object) -> str:
    if isinstance(value, bool):
        return _to_bool_flag(value)
    if value is None:
        return "null"
    if OmegaConf.is_list(value):
        return json.dumps(list(value), separators=(",", ":"))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _flatten_cli_args(prefix: str, value: object, out: list[tuple[str, str]]) -> None:
    if OmegaConf.is_dict(value) or isinstance(value, dict):
        for key, sub_value in value.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _flatten_cli_args(child_prefix, sub_value, out)
        return
    out.append((prefix, _format_cli_value(value)))


def _append_group_args(
    cmd: list[str],
    cfg: DictConfig,
    *,
    cfg_path: str,
    cli_prefix: str,
    skip_keys: set[str] | None = None,
) -> None:
    group = OmegaConf.select(cfg, cfg_path)
    if group is None:
        return
    entries: list[tuple[str, str]] = []
    for key, value in group.items():
        key_str = str(key)
        if skip_keys is not None and key_str in skip_keys:
            continue
        _flatten_cli_args(f"{cli_prefix}.{key_str}", value, entries)
    for key, value in entries:
        cmd.append(f"--{key}={value}")


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

    # 1) Prefer the current interpreter's pip module.
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

    # 3) Last resort: pip executable on PATH (may target user/system site).
    pip_bin = shutil.which("pip", path=env.get("PATH"))
    if pip_bin is not None:
        ok, err = _run_install_command(
            [pip_bin, "install", "--no-deps", "--no-build-isolation", "-e", str(editable_path)],
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
    raw_paths = OmegaConf.select(cfg, "lerobot.install_editables")
    if raw_paths is None:
        legacy_path = OmegaConf.select(cfg, "lerobot.install_editable")
        raw_paths = [] if legacy_path is None else [legacy_path]
    elif not (OmegaConf.is_list(raw_paths) or isinstance(raw_paths, (list, tuple))):
        raise ValueError("lerobot.install_editables must be a list of package paths")

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


def _resolve_libero_benchmark_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("LIBERO package is not importable; cannot bootstrap LIBERO config")

    package_dir = Path(spec.origin).resolve().parent
    benchmark_root = package_dir / "libero"
    if not (benchmark_root / "bddl_files").exists():
        benchmark_root = package_dir
    if not (benchmark_root / "bddl_files").exists():
        raise RuntimeError(f"Could not locate LIBERO benchmark assets under {package_dir}")
    return benchmark_root


def _ensure_libero_config(env: dict[str, str], logger) -> None:
    config_path_raw = env.get("LIBERO_CONFIG_PATH")
    if not config_path_raw:
        return

    config_root = Path(config_path_raw)
    config_file = (
        config_root
        if config_root.suffix.lower() in {".yaml", ".yml"}
        else config_root / "config.yaml"
    )
    if config_file.exists():
        return

    benchmark_root = _resolve_libero_benchmark_root()
    if config_file.parent.name == "libero_config":
        cache_root = config_file.parent.parent
    else:
        cache_root = config_file.parent
    datasets_dir = cache_root / "libero" / "datasets"
    assets_dir = cache_root / "libero" / "assets"

    config_file.parent.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    def _yaml_str(value: Path) -> str:
        return json.dumps(str(value))

    config_body = "\n".join(
        [
            f"benchmark_root: {_yaml_str(benchmark_root)}",
            f"bddl_files: {_yaml_str(benchmark_root / 'bddl_files')}",
            f"init_states: {_yaml_str(benchmark_root / 'init_files')}",
            f"datasets: {_yaml_str(datasets_dir)}",
            f"assets: {_yaml_str(assets_dir)}",
            "",
        ]
    )
    config_file.write_text(config_body, encoding="utf-8")
    logger.info("Bootstrapped LIBERO config: %s", config_file)


def _runtime_cwd_from_cfg(cfg: DictConfig) -> Path:
    run_dir = OmegaConf.select(cfg, "logging.runs_dir")
    if run_dir is None:
        raise ValueError("logging.runs_dir must be set for stage-3 runs")
    cwd = Path(str(run_dir))
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _lerobot_run_command_from_cfg(cfg: DictConfig) -> list[str]:
    train_cmd_raw = OmegaConf.select(cfg, "lerobot.command")

    policy_type = OmegaConf.select(cfg, "lerobot.policy.type")
    policy_repo_id = OmegaConf.select(cfg, "lerobot.policy.repo_id")
    init_mode = OmegaConf.select(cfg, "lerobot.policy.init_mode")
    dataset_repo_id = OmegaConf.select(cfg, "lerobot.dataset.repo_id")
    output_dir = OmegaConf.select(cfg, "lerobot.output_dir")
    job_name = OmegaConf.select(cfg, "lerobot.job_name")
    steps = OmegaConf.select(cfg, "lerobot.steps")
    batch_size = OmegaConf.select(cfg, "lerobot.batch_size")
    grad_accum_steps = OmegaConf.select(cfg, "lerobot.grad_accum_steps")
    num_workers = OmegaConf.select(cfg, "lerobot.num_workers")
    eval_freq = OmegaConf.select(cfg, "lerobot.eval.freq")
    eval_batch_size = OmegaConf.select(cfg, "lerobot.eval.batch_size")
    log_freq = OmegaConf.select(cfg, "lerobot.log_freq")
    save_freq = OmegaConf.select(cfg, "lerobot.save_freq")
    stage2_artifact = OmegaConf.select(cfg, "lerobot.policy.stage2_artifact")

    required = {
        "lerobot.command": train_cmd_raw,
        "lerobot.policy.type": policy_type,
        "lerobot.policy.repo_id": policy_repo_id,
        "lerobot.dataset.repo_id": dataset_repo_id,
        "lerobot.output_dir": output_dir,
        "lerobot.job_name": job_name,
        "lerobot.steps": steps,
        "lerobot.batch_size": batch_size,
        "lerobot.num_workers": num_workers,
        "lerobot.eval.freq": eval_freq,
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
        f"--policy.push_to_hub={_to_bool_flag(OmegaConf.select(cfg, 'lerobot.policy.push_to_hub') is True)}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={int(steps)}",
        f"--batch_size={int(batch_size)}",
        f"--num_workers={int(num_workers)}",
        f"--eval_freq={int(eval_freq)}",
        f"--log_freq={int(log_freq)}",
        f"--save_freq={int(save_freq)}",
        f"--wandb.enable={_to_bool_flag(OmegaConf.select(cfg, 'lerobot.wandb.enable') is True)}",
    ]

    if grad_accum_steps is not None:
        cmd.append(f"--grad_accum_steps={int(grad_accum_steps)}")

    if str(policy_type) == "hlrp_smolvla_shared":
        if init_mode is None:
            raise ValueError("lerobot.policy.init_mode is required for policy_type=hlrp_smolvla_shared")
        cmd.append(f"--policy.init_mode={init_mode}")
        if str(init_mode) == "artifact":
            if stage2_artifact is None:
                raise ValueError(
                    "lerobot.policy.stage2_artifact is required when lerobot.policy.init_mode=artifact"
                )
            cmd.append(f"--policy.stage2_artifact={stage2_artifact}")
        elif str(init_mode) == "scratch":
            if stage2_artifact is not None:
                raise ValueError(
                    "lerobot.policy.stage2_artifact must be null when lerobot.policy.init_mode=scratch"
                )
            cmd.append("--policy.stage2_artifact=null")
        else:
            raise ValueError(
                f"Unsupported lerobot.policy.init_mode={init_mode!r}; expected 'artifact' or 'scratch'"
            )

    if eval_batch_size is not None:
        cmd.append(f"--eval.batch_size={int(eval_batch_size)}")

    policy_device = OmegaConf.select(cfg, "lerobot.policy.device")
    if policy_device is not None:
        cmd.append(f"--policy.device={policy_device}")

    episodes = _episodes_arg(OmegaConf.select(cfg, "lerobot.dataset.episodes"))
    if episodes is not None:
        cmd.append(f"--dataset.episodes={episodes}")

    env_type = OmegaConf.select(cfg, "lerobot.env.type")
    if env_type:
        cmd.append(f"--env.type={env_type}")
    env_task = OmegaConf.select(cfg, "lerobot.env.task")
    if env_task:
        cmd.append(f"--env.task={env_task}")

    _append_group_args(
        cmd,
        cfg,
        cfg_path="lerobot.policy",
        cli_prefix="policy",
        skip_keys={"type", "repo_id", "push_to_hub", "device", "init_mode", "stage2_artifact"},
    )
    _append_group_args(
        cmd,
        cfg,
        cfg_path="lerobot.dataset",
        cli_prefix="dataset",
        skip_keys={"repo_id", "episodes"},
    )
    _append_group_args(
        cmd,
        cfg,
        cfg_path="lerobot.env",
        cli_prefix="env",
        skip_keys={"type", "task"},
    )
    _append_group_args(
        cmd,
        cfg,
        cfg_path="lerobot.eval",
        cli_prefix="eval",
        skip_keys={"freq", "batch_size"},
    )
    _append_group_args(cmd, cfg, cfg_path="lerobot.optimizer", cli_prefix="optimizer")
    _append_group_args(cmd, cfg, cfg_path="lerobot.scheduler", cli_prefix="scheduler")

    use_policy_training_preset = OmegaConf.select(cfg, "lerobot.use_policy_training_preset")
    if use_policy_training_preset is not None:
        cmd.append(f"--use_policy_training_preset={_to_bool_flag(use_policy_training_preset)}")
    return cmd


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger, _ = setup_run_context(
        cfg=cfg,
        workspace_root=workspace_root,
        logger_name="lerobot.training",
    )

    logger.info("=" * 80)
    logger.info("LAPA Stage 3: LeRobot Training")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    env = os.environ.copy()
    packages_path = str(workspace_root / "packages")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        packages_path if not existing_pythonpath else f"{packages_path}:{existing_pythonpath}"
    )
    env_overrides = OmegaConf.select(cfg, "lerobot.shell_env") or {}
    if not OmegaConf.is_dict(env_overrides):
        raise ValueError("lerobot.shell_env must be a mapping of environment variables")
    for k, v in env_overrides.items():
        if v is None:
            continue
        env[str(k)] = str(v)
    _ensure_libero_config(env=env, logger=logger)

    for editable_path in _editable_paths_from_cfg(cfg):
        logger.info("Installing editable package: %s", editable_path)
        _install_editable_package(
            editable_path=editable_path,
            logger=logger,
            cwd=workspace_root,
            env=env,
        )

    cmd = _lerobot_run_command_from_cfg(cfg)
    runtime_cwd = _runtime_cwd_from_cfg(cfg)
    logger.info("Launching LeRobot command:")
    logger.info("  %s", shlex.join(cmd))
    logger.info("  cwd=%s", runtime_cwd)
    subprocess.run(cmd, cwd=str(runtime_cwd), env=env, check=True)
    logger.info("Stage 3 training complete.")


if __name__ == "__main__":
    main()
