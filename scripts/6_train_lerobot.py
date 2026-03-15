#!/usr/bin/env python3
# ruff: noqa: E402
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
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from common.run_context import setup_run_context
from common.libero_runtime import ensure_libero_config


def _to_bool_flag(value: object) -> str:
    return "true" if bool(value) else "false"


def _episodes_arg(value: object) -> str | None:
    if value is None:
        return None
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    s = str(value).strip()
    return s if s else None


def _resolve_dataset_mix_path(cfg: DictConfig) -> Path | None:
    raw_path = OmegaConf.select(cfg, "lerobot.dataset.mix_path")
    if raw_path is None:
        return None
    mix_path = Path(str(raw_path))
    if not mix_path.is_absolute():
        mix_path = workspace_root / mix_path
    mix_path = mix_path.resolve()
    if not mix_path.is_file():
        raise FileNotFoundError(f"Stage-3 dataset mix file not found: {mix_path}")
    return mix_path


def _load_dataset_mix_payload(mix_path: Path) -> dict:
    with open(mix_path) as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected mapping payload in mix file {mix_path}, got {type(payload).__name__}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) == 0:
        raise ValueError(f"Mix file {mix_path} must define one or more sources")
    return payload


def _validate_stage3_mix_for_training_mode(cfg: DictConfig, mix_path: Path) -> None:
    payload = _load_dataset_mix_payload(mix_path)
    mode = str(OmegaConf.select(cfg, "lerobot.policy.stage3_training_mode") or "")
    sources = payload["sources"]
    if not all(isinstance(source, dict) for source in sources):
        raise ValueError(f"Mix file {mix_path} has non-mapping source entries")

    supervision_modes = [str(source.get("supervision")) for source in sources]
    valid_modes = {"latent_only", "multitask"}
    invalid_modes = [mode_name for mode_name in supervision_modes if mode_name not in valid_modes]
    if invalid_modes:
        raise ValueError(
            f"Mix file {mix_path} contains unsupported supervision values: {invalid_modes}"
        )

    has_multitask = any(mode_name == "multitask" for mode_name in supervision_modes)
    has_latent = any(mode_name in {"latent_only", "multitask"} for mode_name in supervision_modes)

    if mode == "action":
        if not has_multitask:
            raise ValueError(
                f"stage3_training_mode='action' requires at least one multitask source in {mix_path}"
            )
        if any(mode_name != "multitask" for mode_name in supervision_modes):
            raise ValueError(
                "stage3_training_mode='action' is incompatible with latent_only mix sources; "
                f"found supervision modes {supervision_modes} in {mix_path}"
            )
    elif mode == "latent":
        if not has_latent:
            raise ValueError(
                f"stage3_training_mode='latent' requires at least one latent-supervised source in {mix_path}"
            )
    elif mode in {"multitask", "alternating"}:
        if not has_multitask:
            raise ValueError(
                f"stage3_training_mode={mode!r} requires at least one multitask source in {mix_path}"
            )
        if not has_latent:
            raise ValueError(
                f"stage3_training_mode={mode!r} requires at least one latent-supervised source in {mix_path}"
            )
    else:
        raise ValueError(f"Unsupported stage3_training_mode={mode!r}")


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


def _primitive_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if OmegaConf.is_dict(value) or OmegaConf.is_list(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _copy_group_config(
    cfg: DictConfig,
    *,
    cfg_path: str,
    skip_keys: set[str] | None = None,
) -> dict[str, object] | None:
    group = OmegaConf.select(cfg, cfg_path)
    if group is None:
        return None
    result: dict[str, object] = {}
    for key, value in group.items():
        key_str = str(key)
        if skip_keys is not None and key_str in skip_keys:
            continue
        result[key_str] = _primitive_value(value)
    return result


def _write_lerobot_train_config(
    cfg: DictConfig,
    *,
    runtime_cwd: Path,
) -> Path:
    resolved_mix_path = _resolve_dataset_mix_path(cfg)
    if resolved_mix_path is not None:
        _validate_stage3_mix_for_training_mode(cfg, resolved_mix_path)

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
    log_freq = OmegaConf.select(cfg, "lerobot.log_freq")
    save_freq = OmegaConf.select(cfg, "lerobot.save_freq")
    stage2_artifact = OmegaConf.select(cfg, "lerobot.policy.stage2_artifact")

    required = {
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

    train_cfg: dict[str, object] = {
        "policy": {
            "type": str(policy_type),
            "repo_id": str(policy_repo_id),
            "push_to_hub": OmegaConf.select(cfg, "lerobot.policy.push_to_hub") is True,
        },
        "dataset": {
            "repo_id": str(dataset_repo_id),
        },
        "output_dir": str(output_dir),
        "job_name": str(job_name),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "eval_freq": int(eval_freq),
        "log_freq": int(log_freq),
        "save_freq": int(save_freq),
        "wandb": {
            "enable": OmegaConf.select(cfg, "lerobot.wandb.enable") is True,
        },
    }

    if grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = int(grad_accum_steps)

    if str(policy_type) == "hlrp_smolvla_shared":
        if init_mode is None:
            raise ValueError(
                "lerobot.policy.init_mode is required for policy_type=hlrp_smolvla_shared"
            )
        train_cfg["policy"]["init_mode"] = str(init_mode)
        if str(init_mode) == "artifact":
            if stage2_artifact is None:
                raise ValueError(
                    "lerobot.policy.stage2_artifact is required when lerobot.policy.init_mode=artifact"
                )
            train_cfg["policy"]["stage2_artifact"] = str(stage2_artifact)
        elif str(init_mode) == "scratch":
            if stage2_artifact is not None:
                raise ValueError(
                    "lerobot.policy.stage2_artifact must be null when lerobot.policy.init_mode=scratch"
                )
            train_cfg["policy"]["stage2_artifact"] = None
        else:
            raise ValueError(
                f"Unsupported lerobot.policy.init_mode={init_mode!r}; expected 'artifact' or 'scratch'"
            )

    eval_batch_size = OmegaConf.select(cfg, "lerobot.eval.batch_size")
    if eval_batch_size is not None:
        train_cfg["eval"] = {"batch_size": int(eval_batch_size)}

    policy_device = OmegaConf.select(cfg, "lerobot.policy.device")
    if policy_device is not None:
        train_cfg["policy"]["device"] = str(policy_device)

    episodes = OmegaConf.select(cfg, "lerobot.dataset.episodes")
    if episodes is not None:
        train_cfg["dataset"]["episodes"] = _primitive_value(episodes)
    if resolved_mix_path is not None:
        train_cfg["dataset"]["mix_path"] = str(resolved_mix_path)

    env_cfg = train_cfg.setdefault("env", {})
    env_type = OmegaConf.select(cfg, "lerobot.env.type")
    if env_type:
        env_cfg["type"] = str(env_type)
    env_task = OmegaConf.select(cfg, "lerobot.env.task")
    if env_task:
        env_cfg["task"] = str(env_task)

    for section_name, cfg_path, skip_keys in (
        (
            "policy",
            "lerobot.policy",
            {
                "type",
                "repo_id",
                "push_to_hub",
                "device",
                "init_mode",
                "stage2_artifact",
            },
        ),
        ("dataset", "lerobot.dataset", {"id", "repo_id", "episodes", "mix_path"}),
        ("env", "lerobot.env", {"type", "task"}),
        ("eval", "lerobot.eval", {"freq", "batch_size"}),
        ("optimizer", "lerobot.optimizer", None),
        ("scheduler", "lerobot.scheduler", None),
        ("wandb", "lerobot.wandb", {"enable"}),
    ):
        section_cfg = _copy_group_config(cfg, cfg_path=cfg_path, skip_keys=skip_keys)
        if section_cfg:
            train_cfg.setdefault(section_name, {}).update(section_cfg)

    use_policy_training_preset = OmegaConf.select(
        cfg, "lerobot.use_policy_training_preset"
    )
    if use_policy_training_preset is not None:
        train_cfg["use_policy_training_preset"] = bool(use_policy_training_preset)

    config_path = runtime_cwd / "lerobot_train_config.json"
    config_path.write_text(json.dumps(train_cfg, indent=2) + "\n")
    return config_path


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


def _runtime_cwd_from_cfg(cfg: DictConfig) -> Path:
    run_dir = OmegaConf.select(cfg, "logging.runs_dir")
    if run_dir is None:
        raise ValueError("logging.runs_dir must be set for stage-3 runs")
    cwd = Path(str(run_dir))
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _lerobot_run_command_from_cfg(cfg: DictConfig) -> list[str]:
    train_cmd_raw = OmegaConf.select(cfg, "lerobot.command")
    if not isinstance(train_cmd_raw, str) or not train_cmd_raw.strip():
        raise ValueError("lerobot.command must be a non-empty string")
    return [train_cmd_raw]


def _resolve_executable(command: str, *, env: dict[str, str]) -> str:
    resolved = shutil.which(command, path=env.get("PATH"))
    return resolved if resolved is not None else command


def _gpu_launch_count_from_cfg(cfg: DictConfig) -> int:
    raw_value = OmegaConf.select(cfg, "cluster.compute.gpus_per_node")
    if raw_value is None:
        return 1
    gpu_count = int(raw_value)
    if gpu_count < 0:
        raise ValueError("cluster.compute.gpus_per_node must be >= 0")
    return gpu_count


def _build_stage3_launch_cmd(
    cfg: DictConfig,
    *,
    env: dict[str, str],
    config_path: Path,
) -> tuple[list[str], str]:
    raw_cmd = _lerobot_run_command_from_cfg(cfg)
    if len(raw_cmd) != 1:
        raise ValueError("lerobot.command must resolve to a single executable name")

    train_executable = _resolve_executable(raw_cmd[0], env=env)
    gpu_count = _gpu_launch_count_from_cfg(cfg)
    node_count = int(OmegaConf.select(cfg, "cluster.compute.num_nodes") or 1)

    base_cmd = [train_executable, "--config_path", str(config_path)]
    if gpu_count <= 1:
        return base_cmd, "single-process"

    if node_count != 1:
        raise ValueError(
            "Stage-3 multi-node launch is not wired yet. "
            "Use cluster.compute.num_nodes=1 for Accelerate-based multi-GPU runs."
        )

    launch_cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        f"--num_processes={gpu_count}",
        "--no_python",
        train_executable,
        "--config_path",
        str(config_path),
    ]
    return launch_cmd, f"accelerate-multi-gpu({gpu_count})"


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
        packages_path
        if not existing_pythonpath
        else f"{packages_path}:{existing_pythonpath}"
    )
    env_overrides = OmegaConf.select(cfg, "lerobot.shell_env") or {}
    if not OmegaConf.is_dict(env_overrides):
        raise ValueError("lerobot.shell_env must be a mapping of environment variables")
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

    runtime_cwd = _runtime_cwd_from_cfg(cfg)
    config_path = _write_lerobot_train_config(cfg, runtime_cwd=runtime_cwd)
    cmd, launch_mode = _build_stage3_launch_cmd(cfg, env=env, config_path=config_path)
    logger.info("Launching LeRobot command:")
    logger.info("  mode=%s", launch_mode)
    logger.info("  %s", shlex.join(cmd))
    logger.info("  cwd=%s", runtime_cwd)
    subprocess.run(cmd, cwd=str(runtime_cwd), env=env, check=True)
    logger.info("Stage 3 training complete.")


if __name__ == "__main__":
    main()
