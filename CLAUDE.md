# CLAUDE.md

## Research Engineering Principles

This is a research codebase. Priorities:
- Keep code clean, DRY, minimal.
- Prefer fail-fast behavior over defensive fallback-heavy logic.
- Avoid excessive input validation; it is acceptable for code to fail so issues surface quickly.
- Keep defaults in Hydra config, not in Python code.
- If a required value is missing from config, prefer explicit failure over silent code defaults.
- Continuously remove unnecessary, bloated, or overly verbose adjacent code when editing.

Comment/documentation policy:
- Do not add process-comments in code.
- Add comments only when they document the final state and help future readers understand the code.
- Document process, rationale, failed alternatives, and migration history in `docs/` instead of inline code comments.

Experiment workflow policy:
- Always prefer an implement-run-analyze iteration loop.
- Make small changes, run intermediate checks, and build incrementally.
- Avoid large untested changes unless unavoidable.
- For experiment tasks, check `docs/` first to avoid repeating already-tested ideas.
- After experiments, document outcomes in `docs/`.

Experiment tracking standard:
- Use metrics-driven comparisons.
- Record each run with config/settings + target metric(s).
- Maintain compact result tables for iterative sweeps (one row per run).
- Use these tables to guide next iterations and decisions.

## Project Overview

Three-stage robot learning system that learns policies from videos without action labels:
1. **Stage 1 (LAM)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Policy)**: Robot policy model predicting latent actions from images + text
3. **Stage 3 (LeRobot finetuning)**: Adapting the policy model to output continuous robot commands

## Repo Map

- `scripts/`: runnable entrypoints (environment setup, stage training, job submission).
- `config/`: Hydra configs (experiments, model/data/training components, cluster presets).
- `packages/`: core Python modules:
  - `packages/lam`: stage-1 latent action training + validation logic.
  - `packages/stage2`: stage-2 robot policy model code.
  - `packages/common`: shared data adapters, logging, utilities.
  - `packages/low_level`: low-level/action-decoder related modules.
- `lerobot_policy_hlrp/`: installable LeRobot policy plugin package used in stage-3 runs.
- `containers/`: container build definitions.
- `docs/`: experiment notes, workflows, and technical documentation.
- `tests/`: pytest-based tests.

## Primary Scripts

- Stage 1 (LAM): `scripts/2_train_stage1_lam.py`
- Stage 2 (Policy): `scripts/4_train_stage2_policy.py`
- Stage 3 (LeRobot): `scripts/6_train_lerobot.py`
- Job submission: `scripts/submit_job.py`

## Repository Locations

- Local: `/mnt/data/workspace/code/high-level-robot-planner`
- Cluster: `~/workspace/code/high-level-robot-planner` (reachable via `ssh ai`)
- DSS root (runs/cache/images): `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay`

## Execution Context

Before deciding how to run anything, check `pwd` and `hostname` to determine if on cluster or workstation.

- If already on cluster and job needs cluster resources, run on cluster.
- If on workstation, decide whether workstation is sufficient; otherwise submit/run on cluster via `ssh ai`.
- Sometimes code is launched directly (without Slurm), both on workstation and on cluster (interactive/debug runs).

### Workstation (Local)

- GPU: RTX 5090 (32 GB VRAM), System RAM: 64 GB
- Use for smaller/short LAM or low-scale training/debug runs.
- Prefer local datasets on workstation:
  - Stage 1/2 LeRobot-v3 datasets via Hydra data configs (for example `data=octo24`)
  - Stage 3 local Libero snapshot: `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/<snapshot>`
- For larger runs, long runs, or shared reproducible jobs, prefer cluster.
- Conda envs:
  - `hlrp` for LAM (stage-1) and Stage-2 policy training
  - `lerobot` for LeRobot (stage-3) training

### Cluster (LRZ/MCML)

- Connect: `ssh ai`
- Never run training on login nodes. Use login nodes only for submission, monitoring, code sync, and lightweight ops. Launch compute allocations/jobs for any actual training.
- Stage 1/2: use LeRobot-v3 datasets and Hugging Face cache paths available on cluster storage.
- Stage 3: Libero can be downloaded at training start by setting `lerobot.dataset.root=null` with `lerobot.dataset.repo_id=HuggingFaceVLA/libero`.
- Check queue: `squeue --me`
- Monitor job: `squeue -j <JOBID> -o "%.18i %.30P %.20j %.8T %.10M %.9l %R"`
- Tail logs: `tail -f <RUN_DIR>/<JOBID>.out` / `.err`
- Job status: `sacct -j <JOBID> --format=JobID,State,ExitCode,Partition,Elapsed -n`

#### Dual-Queue Strategy

Submit same run to both clusters for faster start:
1. Submit to LRZ: `cluster=lrz_x100` (partitions include H100/A100 pools)
2. Submit to MCML: `cluster=mcml_x100` (partitions include H100/A100 pools)
3. Watch both with `squeue`, cancel the slower one with `scancel <JOBID>`

#### Run + Cache Paths

- Runs: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/<timestamp>_<experiment>`
- Cache: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/cache`
- `submit_job.py` mounts repo/runs/cache into container automatically.
- Downloaded/runtime caches to be aware of:
  - `HuggingFaceVLA/libero` dataset cache (`HF_DATASETS_CACHE`)
  - Hugging Face model/checkpoint cache (`HF_HUB_CACHE`)
  - LIBERO assets cache (`~/.cache/libero/assets`)

## Configuration System

Uses Hydra (1.3+) for composable config. Experiments compose components with package paths:

```yaml
defaults:
  - /model@model: lam
  - /data@data: laq_multi_dataset
  - /training@training: stage1_optimizer
  - /cluster@cluster: local_dev
```

Override from CLI:
```bash
python scripts/2_train_stage1_lam.py experiment=stage1_local data.loader.batch_size=32 training.optimizer.lr=5e-5
```

## Submit Workflow

```bash
# Submit job
python scripts/submit_job.py experiment=<experiment_name> [overrides...]

# Dry-run
python scripts/submit_job.py submit.dry_run=true experiment=<experiment_name>

# Sweep (define sweep.params in experiment config)
python scripts/submit_job.py experiment=<sweep_experiment>
```

Local non-Slurm runs (common on workstation and cluster interactive sessions):
```bash
python scripts/2_train_stage1_lam.py experiment=stage1_local
```

See `docs/job_submission.md` for full documentation.

## Testing

```bash
pytest tests/                              # all tests
pytest tests/test_hydra_configs.py -v      # specific file
pytest --cov=packages --cov-report=html tests/  # with coverage
```

Prefer targeted tests for fast iteration before broader test suites. User may call this "pi test".

## Code Quality

```bash
black packages/ scripts/ tests/
ruff check packages/ scripts/ tests/
```

## Dependencies

Python 3.12, PyTorch 2.9.1. Install via:
```bash
conda env create -f environment.yml
conda activate hlrp
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

## Key Architectural Decisions

- **Modular monorepo**: installable packages for tight coupling between stages.
- **Hybrid training framework**: PyTorch Lightning for stages 1 & 3 (DDP), Lightning Fabric for stage 2 (FSDP multi-node).
- **Data pipeline**: LeRobot-v3 source abstraction with weighted multi-dataset sampling.
- **LAM data loading**: unified LeRobot-v3 DataModule path for Stage 1 and Stage 2.
- **Validation**: bucket-strategy binding via `ValidationStrategyCallback`. See config examples in `config/` and implementation in `packages/lam/`.

## Auth

- W&B: enable with `logging.use_wandb=true lerobot.wandb.enable=true`. Auth via `~/.netrc`.
- Hugging Face: `submit_job.py` reads `~/.huggingface/token` and wires `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`.

## Stage 3 Notes

- Main train configs:
  - `config/experiment/stage3_local.yaml` (override `stage3_profile=action_scratch|multitask_scratch`)
  - `config/experiment/stage3_cluster.yaml` (override `stage3_profile=action_scratch|multitask_scratch`)
- Uses LeRobot policy plugin from `lerobot_policy_hlrp/` (editable install).
- Installer fallback order: `python -m pip` → `uv pip` → `pip` (all `--no-deps -e`). Supports different container layouts (with/without pip in active venv).
- For short stage-3 train-only smoke or verification runs, use `cluster.compute.time_limit=00:15:00`.
- For stage-3 eval-enabled smoke runs, use at least `cluster.compute.time_limit=00:45:00`; `00:30:00` is still tight.
`python scripts/submit_job.py experiment=stage3_cluster stage3_profile=action_scratch cluster=lrz_x100 cluster.compute.time_limit=00:45:00 experiment.name=stage3_hlrp_libero_smoke_wandb lerobot.steps=50 lerobot.batch_size=2 lerobot.eval.freq=10 lerobot.eval.batch_size=1 lerobot.eval.n_episodes=1 lerobot.log_freq=10 lerobot.save_freq=1000 logging.use_wandb=true lerobot.wandb.enable=true`

## Containers

- Two separate containers currently:
  - Container A: LAM (stage-1) and Stage-2 policy training
  - Container B: LeRobot (stage-3)
- This is work in progress and may be unified later.
- Stage-1/2 image: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh`
- Stage-3 image: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh`
- Run Enroot import on compute nodes (not login), request enough memory to avoid OOM.
