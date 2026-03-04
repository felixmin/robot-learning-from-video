#!/usr/bin/env python3
"""
Submit training jobs to Slurm by generating sbatch scripts.

This script runs on the login node (no torch required) and generates
sbatch scripts that run inside Enroot containers on compute nodes.

Note:
    This is intended for cluster submissions (Slurm + container execution).
    For local runs, invoke the training script directly, e.g.:
        python scripts/2_train_laq.py experiment=...

Usage:
    # Submit single job
    python scripts/submit_job.py experiment=stage1_octo24_local

    # Override parameters
    python scripts/submit_job.py experiment=stage1_octo24_local training.max_steps=10000

    # Dry run (print script, don't submit)
    python scripts/submit_job.py submit.dry_run=true experiment=stage1_octo24_local

    # Custom resources (Hydra overrides)
    python scripts/submit_job.py cluster.compute.time_limit=01:00:00 cluster.compute.gpus_per_node=2 experiment=laq_full

    # Sweep (reads sweep.params from experiment config)
    python scripts/submit_job.py experiment=laq_lr_sweep
    # Submits one job per parameter combination
"""

import itertools
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure `packages/` is importable when running on cluster login nodes.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "packages"))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.hydra_overrides import normalize_overrides


def parse_sweep_params(cfg) -> dict[str, list[str]]:
    """Extract sweep parameters from sweep.params config.

    Returns dict mapping parameter names to lists of values.
    Example: {'training.optimizer.lr': ['1e-4', '5e-5'], 'seed': ['42', '123']}
    """
    sweep_params = {}

    # Check if sweep.params exists
    if not OmegaConf.select(cfg, "sweep.params"):
        return sweep_params

    params = cfg.sweep.params
    for key, value in params.items():
        # Value can be a string with comma-separated values or a list
        if isinstance(value, str):
            # Parse comma-separated values, strip whitespace
            values = [v.strip() for v in value.split(",")]
        elif isinstance(value, (list, tuple)):
            values = [str(v) for v in value]
        else:
            # Single value
            values = [str(value)]
        sweep_params[key] = values

    return sweep_params


def generate_sweep_combinations(sweep_params: dict[str, list[str]]) -> list[list[str]]:
    """Generate all combinations of sweep parameters as Hydra overrides.

    Returns list of override lists, e.g.:
    [['training.optimizer.lr=1e-4', 'seed=42'], ['training.optimizer.lr=1e-4', 'seed=123'], ...]
    """
    if not sweep_params:
        return [[]]  # Single empty combination (no sweep)

    # Get keys and values in consistent order
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    # Generate Cartesian product
    combinations = []
    for combo in itertools.product(*value_lists):
        overrides = [f"{key}={val}" for key, val in zip(keys, combo)]
        combinations.append(overrides)

    return combinations


def normalize_shell_commands(raw_value, *, field_name: str) -> list[str]:
    """Normalize a shell-command config value into a list of command strings."""

    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        command = raw_value.strip()
        return [command] if command else []

    if OmegaConf.is_list(raw_value) or isinstance(raw_value, (list, tuple)):
        commands: list[str] = []
        for idx, value in enumerate(raw_value):
            if value is None:
                continue
            if not isinstance(value, str):
                raise SystemExit(f"{field_name}[{idx}] must be a string")
            command = value.strip()
            if command:
                commands.append(command)
        return commands

    raise SystemExit(f"{field_name} must be a string or list of strings")


def normalize_paths(raw_value, *, field_name: str) -> list[Path]:
    """Normalize a path config value into a list of absolute paths."""

    raw_paths = normalize_shell_commands(raw_value, field_name=field_name)
    paths: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        paths.append(candidate)
    return paths


def generate_sbatch_script(
    script: str,
    overrides: list,
    partition: str,
    qos: str | None,
    account: str | None,
    gpus: int,
    mem: str | None,
    cpus: int | None,
    container_image: str,
    job_name: str,
    container_mounts: str,
    slurm_logs_dir: Path,
    cache_dir: Path,
    home_dir: Path,
    hf_token_path: Path | None,
    pre_commands: list[str] | None = None,
    python_bin: str = "python",
    *,
    time_limit: str | None = None,
) -> str:
    """Generate sbatch script content."""

    if time_limit is None:
        raise ValueError("Missing time limit (set time_limit or time)")
    if gpus < 0:
        raise ValueError("gpus must be >= 0")
    if cpus is not None and cpus <= 0:
        raise ValueError("cpus must be > 0 when provided")

    # Build the python command with overrides.
    # Quote each override so bash doesn't expand Hydra interpolations like `${now:...}`
    # or `${hydra.job.num}` inside the sbatch script.
    python_args = [python_bin, f"scripts/{script}.py", *overrides]
    python_cmd = " ".join(shlex.quote(str(arg)) for arg in python_args).strip()

    pre_command_block = ""
    if pre_commands:
        pre_lines = [
            "# Run optional pre-commands (e.g., editable policy install).",
            "echo \"Running submit.pre_commands...\"",
        ]
        total = len(pre_commands)
        for idx, cmd in enumerate(pre_commands, start=1):
            pre_lines.append(
                f"echo {shlex.quote(f'[pre_command {idx}/{total}] {cmd}')}"
            )
            pre_lines.append(cmd)
        pre_command_block = "\n".join(pre_lines) + "\n"

    if hf_token_path:
        hf_auth_block = f"""# Hugging Face auth for gated repos (e.g. DINOv3).
if [ -z "${{HF_TOKEN:-}}" ] && [ -f "{hf_token_path}" ]; then
  export HF_TOKEN="$(< "{hf_token_path}")"
fi
if [ -z "${{HUGGINGFACE_HUB_TOKEN:-}}" ] && [ -n "${{HF_TOKEN:-}}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
"""
    else:
        hf_auth_block = """# Hugging Face auth for gated repos (e.g. DINOv3).
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "WARNING: No Hugging Face token found; gated models may fail. Run huggingface-cli login or set HF_TOKEN." >&2
fi
"""

    gres_line = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else ""
    nccl_block = ""
    if gpus > 0:
        nccl_block = """export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
"""

    gpu_info_block = ""
    if gpus > 0:
        gpu_info_block = """# Show GPU info
nvidia-smi
"""
    else:
        gpu_info_block = 'echo "CPU-only job (no GPUs requested)"\n'

    cpus_line = f"#SBATCH --cpus-per-task={cpus}" if cpus is not None else ""
    mem_line = f"#SBATCH --mem={mem}" if mem else ""

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{f"#SBATCH --qos={qos}" if qos else ""}
{f"#SBATCH --account={account}" if account else ""}
{gres_line}
{cpus_line}
{mem_line}
#SBATCH --time={time_limit}
#SBATCH --output={slurm_logs_dir}/%j.out
#SBATCH --error={slurm_logs_dir}/%j.err
#SBATCH --container-image={container_image}
#SBATCH --container-mounts={container_mounts}
#SBATCH --container-workdir={PROJECT_ROOT}

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Environment setup
export PYTHONPATH={PROJECT_ROOT}/packages:${{PYTHONPATH:-}}
{nccl_block}

# Keep auth in the real cluster home, but redirect heavy caches to DSS.
mkdir -p "{cache_dir}/huggingface/hub" "{cache_dir}/huggingface/datasets" "{cache_dir}/huggingface/lerobot" "{cache_dir}/torch" "{cache_dir}/triton"
export HOME="{home_dir}"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="{cache_dir}/huggingface/hub"
export HF_DATASETS_CACHE="{cache_dir}/huggingface/datasets"
export HF_LEROBOT_HOME="{cache_dir}/huggingface/lerobot"
export TORCH_HOME="{cache_dir}/torch"
export TRITON_CACHE_DIR="{cache_dir}/triton"
# Use node-local temp when available (wandb system monitor/GPU stats are sensitive to slow shared TMPDIR).
# Fall back to /tmp if SLURM_TMPDIR isn't set.
# Include SLURM_LOCALID so multi-task jobs don't contend for the same tmp directory.
export TMPDIR="${{SLURM_TMPDIR:-/tmp}}/hlrp-${{SLURM_JOB_ID}}-${{SLURM_LOCALID:-0}}"
mkdir -p "$TMPDIR"

	{hf_auth_block}

{gpu_info_block}

{pre_command_block}

# Run training
{python_cmd}

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
"""
    return script_content


def main():
    overrides = normalize_overrides(sys.argv[1:])

    # Load config to show experiment info and get job name
    config_dir = str(PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Submission defaults (can be overridden via config/experiment/*.yaml or CLI)
    script = OmegaConf.select(cfg, "submit.script") or "2_train_laq"
    dry_run = bool(OmegaConf.select(cfg, "submit.dry_run") or False)

    script_path = PROJECT_ROOT / "scripts" / f"{script}.py"
    if not script_path.exists():
        raise SystemExit(f"Training script not found: {script_path}")

    slurm_enabled = bool(OmegaConf.select(cfg, "cluster.slurm.enabled"))

    # Resolve root + run-group directories.
    logging_root_dir = OmegaConf.select(cfg, "logging.root_dir")
    resolved_logging_root = Path(logging_root_dir) if logging_root_dir else None
    if resolved_logging_root and not resolved_logging_root.is_absolute():
        resolved_logging_root = PROJECT_ROOT / resolved_logging_root
    resolved_logging_root = resolved_logging_root or PROJECT_ROOT

    logging_runs_dir = OmegaConf.select(cfg, "logging.runs_dir")
    if logging_runs_dir:
        runs_dir = Path(logging_runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = resolved_logging_root / runs_dir
    else:
        # Flat structure: runs/{date}_{time}_{experiment}
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = OmegaConf.select(cfg, "experiment.name") or "unknown"
        runs_dir = resolved_logging_root / "runs" / f"{timestamp}_{experiment_name}"

    # Cache dir is stable across runs (by default under logging_root_dir).
    cache_dir = Path(OmegaConf.select(cfg, "submit.cache_dir") or "cache")
    if not cache_dir.is_absolute():
        cache_dir = resolved_logging_root / cache_dir

    pre_commands = normalize_shell_commands(
        OmegaConf.select(cfg, "submit.pre_commands"),
        field_name="submit.pre_commands",
    )
    submit_extra_mounts = normalize_paths(
        OmegaConf.select(cfg, "submit.extra_mounts"),
        field_name="submit.extra_mounts",
    )

    # Check for sweep parameters
    sweep_params = parse_sweep_params(cfg)
    sweep_combinations = generate_sweep_combinations(sweep_params)
    is_sweep = len(sweep_combinations) > 1

    if not slurm_enabled:
        print("=" * 60)
        print("HLRP Local Run (no Slurm)")
        print("=" * 60)
        print(f"\nScript: {script}.py")
        print(f"Experiment: {cfg.experiment.name}")
        print(f"Description: {cfg.experiment.description}")
        print(f"\nPaths:")
        print(f"  Runs dir: {runs_dir}")
        print(f"  Cache dir: {cache_dir}")
        if pre_commands:
            print("  Pre-commands:")
            for pre_cmd in pre_commands:
                print(f"    {pre_cmd}")
        if submit_extra_mounts:
            print("  Extra mounts:")
            for mount_path in submit_extra_mounts:
                print(f"    {mount_path}")
        runs_dir.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if is_sweep:
            print(f"\n🔄 SWEEP MODE: {len(sweep_combinations)} runs")
            print("  (Runs sequentially on the local machine)")

        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = f"{PROJECT_ROOT}/packages:" + base_env.get("PYTHONPATH", "")
        base_env["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
        base_env["HF_DATASETS_CACHE"] = str(cache_dir / "huggingface" / "datasets")
        base_env["HF_LEROBOT_HOME"] = str(cache_dir / "huggingface" / "lerobot")
        base_env["TORCH_HOME"] = str(cache_dir / "torch")
        base_env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")

        for i, sweep_overrides in enumerate(sweep_combinations):
            combined_overrides = normalize_overrides(list(overrides) + sweep_overrides)
            override_str = " ".join(combined_overrides).strip()

            # For sweeps, each job gets its own runs_dir with sweep suffix
            if is_sweep:
                suffix_parts = []
                for override in sweep_overrides:
                    key, val = override.split("=", 1)
                    short_key = key.split(".")[-1]
                    short_val = val.replace("-", "").replace(".", "")[:8]
                    suffix_parts.append(f"{short_key}{short_val}")
                sweep_suffix = "_".join(suffix_parts)
                job_runs_dir = runs_dir.parent / f"{runs_dir.name}_{sweep_suffix}"
            else:
                job_runs_dir = runs_dir

            job_runs_dir.mkdir(parents=True, exist_ok=True)

            cmd_overrides = list(combined_overrides)
            if not any(ov.startswith("logging.runs_dir=") for ov in cmd_overrides):
                cmd_overrides.append(f"logging.runs_dir={job_runs_dir}")
            # Hydra config goes directly in run directory (flat structure).
            if not any(ov.startswith("hydra.run.dir=") for ov in cmd_overrides):
                cmd_overrides.append(f"hydra.run.dir={job_runs_dir}")
            cmd = [sys.executable, str(script_path)] + cmd_overrides

            if dry_run:
                print(f"\n[DRY RUN] {job_runs_dir.name}: {override_str}")
                if pre_commands:
                    print("  pre-commands:")
                    for pre_cmd in pre_commands:
                        print(f"    {pre_cmd}")
                print("  " + " ".join(cmd))
                continue

            print(f"\nRunning {job_runs_dir.name}: {override_str}" if override_str else f"\nRunning {job_runs_dir.name}")
            for pre_cmd in pre_commands:
                print(f"  Running pre-command: {pre_cmd}")
                subprocess.run(
                    pre_cmd,
                    cwd=str(PROJECT_ROOT),
                    env=base_env,
                    shell=True,
                    check=True,
                )
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=base_env, check=True)

        return

    # Resolve Slurm/compute settings from Hydra config (override via CLI, e.g. cluster.compute.time_limit=...).
    partition = OmegaConf.select(cfg, "cluster.slurm.partition") or "mcml-hgx-h100-94x4"
    qos = OmegaConf.select(cfg, "cluster.slurm.qos")
    account = OmegaConf.select(cfg, "cluster.slurm.account")
    gpus_value = OmegaConf.select(cfg, "cluster.compute.gpus_per_node")
    gpus = 1 if gpus_value is None else int(gpus_value)
    if gpus < 0:
        raise SystemExit("cluster.compute.gpus_per_node must be >= 0")
    cpus_value = OmegaConf.select(cfg, "cluster.compute.cpus_per_task")
    cpus = int(cpus_value) if cpus_value is not None else None
    if cpus is not None and cpus <= 0:
        raise SystemExit("cluster.compute.cpus_per_task must be > 0 when set")
    time_limit = OmegaConf.select(cfg, "cluster.compute.time_limit") or "24:00:00"

    # Container image is required for Slurm submissions.
    container_image = OmegaConf.select(cfg, "cluster.container.image")
    if not container_image:
        cluster_name = OmegaConf.select(cfg, "cluster.name") or "<unknown>"
        raise SystemExit(
            "Missing container image. Set `cluster.container.image` in the cluster config "
            f"(cluster={cluster_name})."
        )
    container_image_path = Path(str(container_image))
    if not container_image_path.exists():
        raise SystemExit(
            "Container image not found: "
            f"{container_image_path}\n"
            "Set `cluster.container.image=/path/to/container.sqsh` (or update your cluster config)."
        )
    container_python_bin = str(OmegaConf.select(cfg, "cluster.container.python_bin") or "python")

    mem_value = OmegaConf.select(cfg, "cluster.compute.mem_gb")
    if mem_value is None:
        mem = None
    else:
        mem_gb = int(mem_value)
        if mem_gb <= 0:
            raise SystemExit("cluster.compute.mem_gb must be > 0 when set")
        mem = f"{mem_gb}G"

    # Ensure directories exist on the shared filesystem.
    # For single jobs, runs_dir is the job directory.
    # For sweeps, job directories are created in the loop (siblings of runs_dir).
    runs_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    home_dir = Path.home().resolve()

    # Hugging Face auth token path (if present) for gated model downloads inside the container.
    hf_token_path = None
    candidate = home_dir / ".huggingface" / "token"
    if candidate.exists():
        hf_token_path = candidate

    # External artifact paths that must be visible inside the container (e.g., LAQ checkpoint).
    extra_mounts: list[Path] = []
    laq_ckpt = OmegaConf.select(cfg, "model.laq.checkpoint")
    if laq_ckpt:
        laq_ckpt_path = Path(str(laq_ckpt))
        if laq_ckpt_path.is_absolute():
            extra_mounts.append(laq_ckpt_path.parent)
    resume_ckpt = OmegaConf.select(cfg, "training.resume_from_checkpoint")
    if resume_ckpt:
        resume_ckpt_path = Path(str(resume_ckpt))
        if resume_ckpt_path.is_absolute():
            extra_mounts.append(resume_ckpt_path.parent)
    weights_ckpt = OmegaConf.select(cfg, "training.load_weights_from")
    if weights_ckpt:
        weights_ckpt_path = Path(str(weights_ckpt))
        if weights_ckpt_path.is_absolute():
            extra_mounts.append(weights_ckpt_path.parent)

    # Build container mounts: always mount the project root, plus any external run/cache roots.
    # Mount runs_dir.parent to include all sweep job directories (which are siblings).
    for mount_path in submit_extra_mounts:
        if not mount_path.exists():
            print(
                "WARNING: submit.extra_mounts path does not exist on the submit host; "
                f"skipping container mount: {mount_path}"
            )
            continue
        extra_mounts.append(mount_path)

    mount_roots: list[Path] = [PROJECT_ROOT, runs_dir.parent, cache_dir, home_dir, *extra_mounts]

    # Ensure unique mount roots while preserving order.
    seen: set[Path] = set()
    unique_mounts: list[Path] = []
    for p in mount_roots:
        if p in seen:
            continue
        seen.add(p)
        unique_mounts.append(p)

    container_mounts = ",".join(f"{p}:{p}" for p in unique_mounts)

    print("=" * 60)
    print("HLRP Job Submission")
    print("=" * 60)
    print(f"\nScript: {script}.py")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")

    if is_sweep:
        print(f"\n🔄 SWEEP MODE: {len(sweep_combinations)} jobs")
        print(f"  Sweep parameters:")
        for param, values in sweep_params.items():
            print(f"    {param}: {values}")

    print(f"\nSlurm settings:")
    print(f"  Partition: {partition}")
    if qos:
        print(f"  QoS: {qos}")
    if account:
        print(f"  Account: {account}")
    if gpus == 0:
        print("  GPUs: 0 (CPU-only)")
    else:
        print(f"  GPUs: {gpus}")
    print(f"  Time: {time_limit}")
    print(f"  Memory: {mem}" if mem else "  Memory: cluster default (unset)")
    print(f"  CPUs: {cpus}" if cpus is not None else "  CPUs: cluster default (unset)")
    print(f"  Container: {container_image}")
    print(f"  Runs dir: {runs_dir}")
    print(f"  Cache dir: {cache_dir}")
    if pre_commands:
        print("  Pre-commands:")
        for pre_cmd in pre_commands:
            print(f"    {pre_cmd}")
    if submit_extra_mounts:
        print("  Extra mounts:")
        for mount_path in submit_extra_mounts:
            print(f"    {mount_path}")

    # Submit jobs for each sweep combination
    submitted_jobs = []

    for i, sweep_overrides in enumerate(sweep_combinations):
        # Combine base overrides with sweep overrides
        # Sweep overrides come last to take precedence
        combined_overrides = normalize_overrides(list(overrides) + sweep_overrides)

        # For sweeps, each job gets its own runs_dir with sweep suffix
        if is_sweep:
            # Create a short suffix from sweep params (e.g., "lr1e-4_seed42")
            suffix_parts = []
            for override in sweep_overrides:
                key, val = override.split("=", 1)
                # Use last part of key (e.g., "lr" from "training.optimizer.lr")
                short_key = key.split(".")[-1]
                # Shorten value if needed
                short_val = val.replace("-", "").replace(".", "")[:8]
                suffix_parts.append(f"{short_key}{short_val}")
            sweep_suffix = "_".join(suffix_parts)
            job_runs_dir = runs_dir.parent / f"{runs_dir.name}_{sweep_suffix}"
            job_name = f"hlrp_{cfg.experiment.name}_{sweep_suffix}"
        else:
            job_runs_dir = runs_dir
            job_name = f"hlrp_{cfg.experiment.name}"

        # Ensure job directory exists
        job_runs_dir.mkdir(parents=True, exist_ok=True)

        # Force run outputs into this job's run directory unless user overrides.
        if not any(ov.startswith("logging.runs_dir=") for ov in combined_overrides):
            combined_overrides.append(f"logging.runs_dir={job_runs_dir}")

        # Hydra config goes directly in run directory (flat structure).
        if not any(ov.startswith("hydra.run.dir=") for ov in combined_overrides):
            combined_overrides.append(f"hydra.run.dir={job_runs_dir}")
        if not any(ov.startswith("hydra.sweep.dir=") for ov in combined_overrides):
            combined_overrides.append(f"hydra.sweep.dir={job_runs_dir}")
        if not any(ov.startswith("hydra.sweep.subdir=") for ov in combined_overrides):
            combined_overrides.append("hydra.sweep.subdir=.")

        # Generate sbatch script
        sbatch_content = generate_sbatch_script(
            script=script,
            overrides=combined_overrides,
            partition=partition,
            qos=qos,
            account=account,
            gpus=gpus,
            time_limit=time_limit,
            mem=mem,
            cpus=cpus,
            container_image=container_image,
            job_name=job_name,
            container_mounts=container_mounts,
            slurm_logs_dir=job_runs_dir,
            cache_dir=cache_dir,
            home_dir=home_dir,
            hf_token_path=hf_token_path,
            pre_commands=pre_commands,
            python_bin=container_python_bin,
        )

        if dry_run:
            if is_sweep:
                print(f"\n[DRY RUN] Job {i+1}/{len(sweep_combinations)}: {sweep_overrides}")
            print("-" * 60)
            print(sbatch_content)
            print("-" * 60)
            continue

        # Write and submit sbatch script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as f:
            f.write(sbatch_content)
            sbatch_file = f.name

        try:
            if is_sweep:
                print(f"\nSubmitting job {i+1}/{len(sweep_combinations)}: {sweep_overrides}")
            else:
                print("\nSubmitting job...")

            result = subprocess.run(
                ["sbatch", sbatch_file],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from output like "Submitted batch job 12345"
            output = result.stdout.strip()
            job_id = output.split()[-1] if output else "unknown"
            submitted_jobs.append((job_id, job_runs_dir, sweep_overrides))

            print(f"  {output}")

        except subprocess.CalledProcessError as e:
            print(f"\nError submitting job: {e.stderr}")
            raise
        finally:
            # Clean up temp file
            os.unlink(sbatch_file)

    # Print summary
    if not dry_run and submitted_jobs:
        print("\n" + "=" * 60)
        print(f"Submitted {len(submitted_jobs)} job(s)")
        print("=" * 60)
        print("\nMonitor with:")
        print("  squeue --me")
        if len(submitted_jobs) == 1:
            job_id, job_dir, _ = submitted_jobs[0]
            print(f"  tail -f {job_dir}/{job_id}.out")
        else:
            print("\nJobs:")
            for job_id, job_dir, sweep_ov in submitted_jobs:
                print(f"  {job_id}: {job_dir.name}")
                print(f"    tail -f {job_dir}/{job_id}.out")


if __name__ == "__main__":
    main()
