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
1. **Stage 1 (LAQ)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation/VLA)**: Vision-Language model predicting latent actions from images + text
3. **Stage 3 (LeRobot finetuning)**: Adapting the foundation model to output continuous robot commands

## Repo Map

- `scripts/`: runnable entrypoints (environment setup, stage training, job submission).
- `config/`: Hydra configs (experiments, model/data/training components, cluster presets).
- `packages/`: core Python modules:
  - `packages/laq`: stage-1 latent action training + validation logic.
  - `packages/foundation`: stage-2 foundation/VLA model code.
  - `packages/common`: shared data adapters, logging, utilities.
  - `packages/low_level`: low-level/action-decoder related modules.
- `lerobot_policy_hlrp/`: installable LeRobot policy plugin package used in stage-3 runs.
- `containers/`: container build definitions.
- `docs/`: experiment notes, workflows, and technical documentation.
- `tests/`: pytest-based tests.

## Primary Scripts

- Stage 1 (LAQ): `scripts/2_train_laq.py`
- Stage 2 (VLA): `scripts/4_train_foundation.py`
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
- Use for smaller/short LAQ or low-scale training/debug runs.
- Prefer local datasets on workstation:
  - Stage 1/2 local OXE shards: `/mnt/data/oxe` (via `data=oxe_local_indexed`)
  - Stage 3 local Libero snapshot: `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/<snapshot>`
- For larger runs, long runs, or shared reproducible jobs, prefer cluster.
- Conda envs:
  - `hlrp` for LAQ (stage-1) and VLA (stage-2) training
  - `lerobot` for LeRobot (stage-3) training

### Cluster (LRZ/MCML)

- Connect: `ssh ai`
- Never run training on login nodes. Use login nodes only for submission, monitoring, code sync, and lightweight ops. Launch compute allocations/jobs for any actual training.
- Stage 1/2: use dataset paths available on cluster storage (no automatic OXE download by training scripts).
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
  - /model@model: laq
  - /data@data: laq_multi_dataset
  - /training@training: laq_optimizer
  - /cluster@cluster: local_dev
```

Override from CLI:
```bash
python scripts/2_train_laq.py experiment=stage1_laq_oxe_local data.loader.batch_size=32 training.optimizer.lr=5e-5
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
python scripts/2_train_laq.py experiment=stage1_laq_oxe_local
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
- **Data pipeline**: WebDataset with TAR shards for GPFS-optimized sequential reads.
- **LAQ data loading**: two modes — local multi-dataset (`LAQDataModule`) and OXE streaming (`OXEDataModule`, auto-detected by `dataset_name` field).
- **Validation**: bucket-strategy binding via `ValidationStrategyCallback`. See config examples in `config/` and implementation in `packages/laq/`.

## Auth

- W&B: enable with `logging.use_wandb=true lerobot.wandb.enable=true`. Auth via `~/.netrc`.
- Hugging Face: `submit_job.py` reads `~/.huggingface/token` and wires `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`.

## Stage 3 Notes

- Main train configs:
  - `config/experiment/stage3_hlrp_libero_action_scratch.yaml`
  - `config/experiment/stage3_hlrp_libero_multitask_scratch.yaml`
- Uses LeRobot policy plugin from `lerobot_policy_hlrp/` (editable install).
- Installer fallback order: `python -m pip` → `uv pip` → `pip` (all `--no-deps -e`). Supports different container layouts (with/without pip in active venv).
- For short smoke or verification runs, always add `cluster.compute.time_limit=00:15:00` so the scheduler does not treat them like long jobs.
- Example smoke command: `python scripts/submit_job.py experiment=stage3_hlrp_libero_action_scratch cluster=lrz_x100 cluster.compute.time_limit=00:15:00 experiment.name=stage3_hlrp_libero_smoke_retry lerobot.steps=50 lerobot.batch_size=2 lerobot.eval.freq=10 lerobot.log_freq=10 lerobot.save_freq=1000`
- With W&B: `python scripts/submit_job.py experiment=stage3_hlrp_libero_action_scratch cluster=lrz_x100 cluster.compute.time_limit=00:15:00 experiment.name=stage3_hlrp_libero_smoke_wandb lerobot.steps=50 lerobot.batch_size=2 lerobot.eval.freq=10 lerobot.log_freq=10 lerobot.save_freq=1000 logging.use_wandb=true lerobot.wandb.enable=true`

## Containers

- Two separate containers currently:
  - Container A: LAQ (stage-1) and VLA (stage-2)
  - Container B: LeRobot (stage-3)
- This is work in progress and may be unified later.
- Stage-1/2 image: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh`
- Stage-3 image: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh`
- Run Enroot import on compute nodes (not login), request enough memory to avoid OOM.
