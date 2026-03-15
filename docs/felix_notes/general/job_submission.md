# Job Submission Guide

**Date:** 2025-12-24
**Status:** Working

This guide explains how to submit training jobs to the LRZ cluster using `scripts/submit_job.py`.

---

## Quick Start

```bash
# Submit a single job
python scripts/submit_job.py experiment=laq_oxe_local cluster=lrz_h100

# Submit with custom time limit
python scripts/submit_job.py cluster.compute.time_limit=04:00:00 experiment=laq_oxe_cluster

# Dry run (preview without submitting)
python scripts/submit_job.py submit.dry_run=true experiment=laq_oxe_local cluster=lrz_h100

# Submit a sweep (multiple jobs)
python scripts/submit_job.py experiment=laq_oxe_local_sweep cluster=lrz_h100
```

---

## How It Works

The `submit_job.py` script:

1. **Runs on the login node** (no GPU/torch required)
2. **Loads Hydra config** to get experiment info and sweep parameters
3. **Generates sbatch scripts** with container directives
4. **Submits via `sbatch`** command

This approach bypasses Hydra's submitit launcher (which requires torch imports) by generating sbatch scripts directly.

## Local Runs (No Slurm)

For local development/training, run the training script directly (no `sbatch`, no container directives):

```bash
python scripts/2_train_stage1_lam.py experiment=laq_oxe_local
```

Use `scripts/submit_job.py` only when you want Slurm scheduling + enroot container execution on the cluster.

---

## Single Job Submission

### Basic Usage

```bash
python scripts/submit_job.py experiment=laq_oxe_local cluster=lrz_h100
```

### With Overrides

```bash
python scripts/submit_job.py experiment=laq_oxe_cluster \
    training.epochs=50 \
    data.loader.batch_size=64
```

### Custom Resources

```bash
python scripts/submit_job.py \
    cluster.compute.gpus_per_node=4 \
    cluster.compute.time_limit=08:00:00 \
    cluster.compute.mem_gb=128 \
    cluster.compute.cpus_per_task=16 \
    experiment=laq_oxe_cluster
```

### Different Training Script

```bash
# Run foundation training instead of LAQ
python scripts/submit_job.py experiment=vla_smol_flow_shared
```

---

## Sweep Submission

Sweeps submit multiple jobs with different parameter combinations.

### Define Sweep in Experiment Config

```yaml
# config/experiment/laq_oxe_local_sweep.yaml
# @package _global_

defaults:
  - /model@model: lam
  - /data@data: oxe_local_indexed
  - /training@training: stage1_optimizer
  - /cluster@cluster: local_dev

# Sweep parameters - comma-separated values
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5, 1e-5
    seed: 42, 123

experiment:
  name: laq_oxe_local_sweep
  description: "Learning rate sweep for LAQ"

# ... rest of config
```

### Submit Sweep

```bash
python scripts/submit_job.py experiment=laq_oxe_local_sweep cluster=lrz_h100
```

**Output:**
```
🔄 SWEEP MODE: 6 jobs
  Sweep parameters:
    training.optimizer.lr: ['1e-4', '5e-5', '1e-5']
    seed: ['42', '123']

Submitting job 1/6: ['training.optimizer.lr=1e-4', 'seed=42']
  Submitted batch job 5423108
Submitting job 2/6: ['training.optimizer.lr=1e-4', 'seed=123']
  Submitted batch job 5423109
...

Job IDs:
  5423108: ['training.optimizer.lr=1e-4', 'seed=42']
  5423109: ['training.optimizer.lr=1e-4', 'seed=123']
  ...
```

### How Sweeps Work

1. **Parse `sweep.params`** from experiment config
2. **Split comma-separated values** into lists
3. **Generate Cartesian product** of all combinations
4. **Submit separate job** for each combination
5. **Unique job names** include parameter values (e.g., `hlrp_laq_oxe_local_sweep_lr1e4_seed42`)

### Sweep Parameter Syntax

```yaml
sweep:
  params:
    # Multiple values (comma-separated)
    training.optimizer.lr: 1e-4, 5e-5, 1e-5

    # Seeds for reproducibility
    seed: 42, 123, 456

    # Any config path works
    model.codebook_size: 8, 16, 32
    data.loader.batch_size: 32, 64
```

**Note:** Uses `sweep.params` (not `hydra.sweeper.params`) because Hydra's internal config isn't accessible via `compose()`.

---

## CLI Options

This script is a pure Hydra CLI. Common overrides:

- `submit.dry_run=true` (print sbatch scripts, don’t submit)
- `submit.script=4_train_stage2_policy` (override which `scripts/*.py` entrypoint runs)
- `submit.pre_commands=[...]` (run shell commands before training starts)
- `cluster=<name>` (select cluster config)
- `cluster.compute.gpus_per_node=...`
- `cluster.compute.cpus_per_task=...`
- `cluster.compute.mem_gb=...`
- `cluster.compute.time_limit=HH:MM:SS`
- `cluster.slurm.partition=...`, `cluster.slurm.qos=...`, `cluster.slurm.account=...`
- `cluster.container.image=/path/to/container.sqsh`

If `cluster.compute.cpus_per_task` or `cluster.compute.mem_gb` are not set, `submit_job.py`
does not emit `#SBATCH --cpus-per-task` / `#SBATCH --mem`, so Slurm/cluster defaults are used.

---

## Installing Local Policy Plugins Per Job

When policy code changes frequently and should stay outside the container image, run an editable install at job start:

```bash
python scripts/submit_job.py \
    cluster=lrz_h100 \
    experiment=vla_smol \
    'submit.pre_commands=["pip install -e /dss/.../high-level-robot-planner/lerobot_policy_hlrp"]'
```

This executes `pip install -e ...` inside the job container before the training command, so `lerobot`
policy discovery can see the plugin package for that run.

For multiple setup steps, pass multiple commands:

```bash
python scripts/submit_job.py \
    cluster=lrz_h100 \
    experiment=vla_smol \
    'submit.pre_commands=["pip install --no-deps -e /dss/.../high-level-robot-planner/lerobot","pip install --no-deps -e /dss/.../high-level-robot-planner/lerobot_policy_hlrp"]'
```

---

## Container Configuration

The container image path is required. In the normal operator workflow it comes
from `config/user_config/local.yaml`, which points at a dated imported unified
`.sqsh` under DSS `enroot/`. One-off Hydra overrides can still replace it for a
single submission.

```yaml
# config/user_config/local.yaml
cluster:
  container:
    image: /dss/.../enroot/hlrp_unified_cu128_imported_<timestamp>.sqsh
```

### Override Container

```bash
# Via Hydra override
python scripts/submit_job.py cluster.container.image=/path/to/custom.sqsh experiment=laq_oxe_local
```

---

## Caching (Pretrained Weights)

The generated sbatch script sets Hugging Face and torch/torchvision cache env vars so
pretrained weights persist across jobs. Configure the base directory with `submit.cache_dir`
(default: `cache/`).

If you use gated Hugging Face models, make sure the job can authenticate:

- Run `huggingface-cli login` once on the cluster (stores a token under your home directory), or
- Export `HF_TOKEN=...` in your environment before submitting.

---

## Monitoring Jobs

### Check Queue Status

```bash
ssh ai 'squeue --me'
```

**Output:**
```
JOBID PARTITION     NAME     USER ST  TIME  NODES NODELIST(REASON)
5423108 mcml-hgx- hlrp_laq go98qik2  R  5:23      1 mcml-hgx-h100-006
5423109 mcml-hgx- hlrp_laq go98qik2 PD  0:00      1 (Priority)
```

Status codes:
- `R` = Running
- `PD` = Pending (waiting for resources)

### View Job Output

```bash
# Live output
ssh ai 'tail -f /dss/.../runs/slurm/5423108.out'

# Error log
ssh ai 'cat /dss/.../runs/slurm/5423108.err'
```

### Cancel Jobs

```bash
# Cancel single job
ssh ai 'scancel 5423108'

# Cancel all your jobs
ssh ai 'scancel --me'
```

---

## Generated sbatch Script

Each job generates a script like:

```bash
#!/bin/bash
#SBATCH --job-name=hlrp_laq_oxe_local_sweep_lr1e4_seed42
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16  # Optional; omitted when unset
#SBATCH --mem=128G          # Optional; omitted when unset
#SBATCH --time=15:00:00
#SBATCH --output=/dss/.../runs/slurm/%j.out
#SBATCH --error=/dss/.../runs/slurm/%j.err
#SBATCH --container-image=/dss/.../enroot/hlrp_unified_cu128_imported_<timestamp>.sqsh
#SBATCH --container-mounts=/dss/.../high-level-robot-planner:/dss/.../high-level-robot-planner
#SBATCH --container-workdir=/dss/.../high-level-robot-planner

# Environment setup
export PYTHONPATH=/dss/.../packages:$PYTHONPATH
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

nvidia-smi

# Run training with overrides
python scripts/2_train_stage1_lam.py experiment=laq_oxe_local_sweep cluster=lrz_h100 training.optimizer.lr=1e-4 seed=42
```

---

## Example Sweep Configs

### Learning Rate Sweep

```yaml
# config/experiment/laq_oxe_local_sweep.yaml
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5, 1e-5
    seed: 42, 123
```
Submits 6 jobs (3 LRs × 2 seeds).

### Model Architecture Sweep

```yaml
# config/experiment/laq_arch_sweep.yaml
sweep:
  params:
    model.codebook_size: 8, 16, 32, 64
    model.code_dim: 32, 64
```
Submits 8 jobs (4 codebook sizes × 2 code dims).

### Dataset Comparison

```yaml
# config/experiment/laq_dataset_sweep.yaml
sweep:
  params:
    data.dataset_name: language_table, bridge
    seed: 42, 123, 456
```
Submits 6 jobs (2 datasets × 3 seeds).

---

## Troubleshooting

### "Container image not found"

Update `config/user_config/local.yaml` to a valid imported `.sqsh` (or override
with `cluster.container.image=/path/to/container.sqsh` for a one-off run).

### Job stuck in "Priority" state

Normal on shared clusters. Jobs wait for resources. Check estimated start:
```bash
ssh ai 'squeue --me --start'
```

### "No module named 'torch'" on login node

This is expected! The submit script doesn't need torch - it only parses Hydra configs. Training runs inside the container on compute nodes.

### Sweep parameters not detected

Make sure you use `sweep.params` (not `hydra.sweeper.params`):
```yaml
# Correct
sweep:
  params:
    training.optimizer.lr: 1e-4, 5e-5

# Won't work (Hydra internal config)
hydra:
  sweeper:
    params:
      training.optimizer.lr: 1e-4, 5e-5
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/submit_job.py` | Main submission script |
| `config/experiment/*.yaml` | Experiment configs (can include sweep.params) |
| `runs/slurm/*.out` | Job stdout logs |
| `runs/slurm/*.err` | Job stderr logs |

---

## Design Decisions

### Why Not Hydra's Submitit Launcher?

We use a custom `submit_job.py` script instead of Hydra's built-in submitit launcher for these reasons:

**1. Training scripts import torch at module level**
```python
# scripts/2_train_stage1_lam.py
import torch  # ← Fails on login node (no torch installed)

@hydra.main(...)
def main(cfg):
    ...
```

Hydra's submitit launcher (`-m` flag) executes the script on the login node to parse the config, then pickles the function. This fails because torch isn't installed on the login node.

**Solution comparison:**
- **Hydra submitit:** Requires refactoring all scripts to lazy-import torch inside `main()`
- **Our approach:** Parse configs separately, generate sbatch scripts directly

**2. Container integration is simpler**

- **Hydra submitit:** Requires wrapper scripts or pickle compatibility inside container
- **Our approach:** Uses SBATCH container directives natively (`#SBATCH --container-image=...`)

**3. No pickle mechanism needed**

- **Hydra submitit:** Pickles Python functions, requires submitit installed in container
- **Our approach:** Generates shell scripts that call `python scripts/X.py` with overrides

### Comparison

| Feature | Hydra Submitit | submit_job.py |
|---------|---------------|---------------|
| Requires torch on login node | Yes | No |
| Sweep support | Built-in (`-m` flag) | Via `sweep.params` |
| Container support | Needs wrapper | Native SBATCH directives |
| Pickle compatibility | Required | Not needed |
| Config syntax | `hydra.sweeper.params` | `sweep.params` |
| Refactoring required | Move imports to `main()` | None |

### Alternative Approaches Considered

**1. Pip install at runtime:** Install missing deps when job starts
- **Issue:** PyTorch version conflicts between base container and pytorch-lightning
- **When it works:** If using container with compatible PyTorch version

**2. Submitit with SlurmExecutor:** Use submitit directly without Hydra
- **Issue:** Requires submitit installed in container (base containers don't have it)
- **When it works:** If building custom container with submitit included

**3. Hydra multirun with custom launcher:** Write custom Hydra launcher plugin
- **Issue:** Complex, reinvents what sbatch already does
- **When it works:** If you need Hydra's advanced sweep algorithms (Bayesian optimization, etc.)

Our approach (direct sbatch generation) is the simplest solution that works with minimal changes to existing code and containers.
