# HLRP Operational Notes

## Scope
This file captures the practical workflow used in this repo for:
- Cluster access and job monitoring
- Submitting training jobs via `scripts/submit_job.py`
- Multi-stage training pipeline (`LAM` -> `Stage2 Policy` -> `LeRobot`)
- Dual-queue strategy (`lrz_x100` and `mcml_x100`) with cancel-on-first-start

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

## Cluster Safety Rules (Strict)
When operating on cluster systems:
- Never delete anything. No exceptions.
- Never run destructive cleanup/delete commands.
- If disk space is exhausted or near limit: stop immediately and ask what to do next.
- Only operate within:
  - user home directory
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay`
- Do not modify anything outside those locations.
- Inside allowed locations, code/file edits are allowed.
- Small-scale file operations (e.g., targeted rename/move) are allowed when needed and low-risk.
- No deletion of files/directories, and no large-scale filesystem operations.

## Repository Locations
- Local repo path:
  - `/mnt/data/workspace/code/high-level-robot-planner`
- Cluster repo path:
  - `~/workspace/code/high-level-robot-planner` (reachable via `ssh ai`)
- Main DSS root used for runs/cache/images:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay`

## Repo Map (Brief)
- `scripts/`: runnable entrypoints (environment setup, stage training, job submission).
- `config/`: Hydra configs (experiments, model/data/training components, cluster presets, user overrides).
- `packages/`: core Python modules:
  - `packages/lam`: stage-1 latent action training + validation logic.
  - `packages/stage2`: stage-2 robot policy model code.
  - `packages/common`: shared data adapters, logging, utilities.
  - `packages/low_level`: low-level/action-decoder related modules.
- `lerobot_policy_hlrp/`: installable LeRobot policy plugin package used in stage-3 runs.
- `containers/`: container build definitions.
- `docs/`: experiment notes, workflows, and technical documentation.
- `tests/`: pytest-based tests.

## Execution Context (Always Check First)
Before deciding how to run anything, first check:
- `pwd`
- `hostname`
- whether you are on cluster or workstation

Decision rule:
- If already on cluster and job needs cluster resources, run on cluster.
- If on workstation, decide whether workstation is sufficient; otherwise submit/run on cluster via `ssh ai`.
- Sometimes code is launched directly from code folders (without Slurm submit flow), both on workstation and on cluster (typically interactive/debug runs).

Workstation capability reference (for local non-Slurm runs):
- GPU: RTX 5090 (32 GB VRAM)
- System RAM: 64 GB
- Use local workstation for smaller/short LAM or low-scale training/debug runs when resources are sufficient.
- Prefer local datasets on workstation:
  - Stage 1/2 LeRobot-v3 datasets via Hydra data configs (for example `data=octo24`)
  - Stage 3 local Libero snapshot root: `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/<snapshot>`
- "Not representative for cluster behavior" is not, by itself, a reason to reject local training runs when the goal is local training.
- For larger runs, long runs, or shared reproducible jobs, prefer cluster.
- Conda env selection on workstation:
  - Use `hlrp` conda env for latent action model (LAM) and Stage-2 policy training.
  - Use `lerobot` conda env for LeRobot/stage-3 training.

## Primary Scripts
- Stage 1 (Latent Action Model pretraining):
  - `scripts/2_train_stage1_lam.py`
- Stage 2 (Policy training):
  - `scripts/4_train_stage2_policy.py`
- Stage 3 (LeRobot fine-tune/eval integration):
  - `scripts/6_train_lerobot.py`
- Unified Slurm submission entrypoint:
  - `scripts/submit_job.py`

## Stage 3 (LeRobot) Config
- Main train configs:
  - `config/experiment/stage3_local.yaml` (override `stage3_profile=action_scratch|multitask_scratch`)
  - `config/experiment/stage3_cluster.yaml` (override `stage3_profile=action_scratch|multitask_scratch`)
- Uses LeRobot policy plugin editable install from:
  - `lerobot_policy_hlrp`

## Dataset Source and Download Behavior
- Workstation/local runs:
  - Stage 1/2 use LeRobot-v3 datasets via Hydra data config (for example `data=octo24`).
  - Stage 3 typically uses a local Libero snapshot via `lerobot.dataset.root=/mnt/data/workspace/hflibero/...`.
- Cluster runs:
  - Stage 1/2 use LeRobot-v3 datasets and Hugging Face cache paths on cluster storage.
  - Stage 3 can download Libero at train start by setting `lerobot.dataset.root=null` and using `lerobot.dataset.repo_id=HuggingFaceVLA/libero`.
- Runtime-downloaded datasets/assets that should be documented:
  - `HuggingFaceVLA/libero` dataset cache (`HF_DATASETS_CACHE`).
  - LIBERO environment assets cache (`~/.cache/libero/assets`) if missing.
  - Hugging Face model/checkpoint cache (`HF_HUB_CACHE`) for model weights used by Stage 2/3 and LAM dependencies.

## Cluster Access + Basic Operations
- Connect:
  - `ssh ai`
- Go to repo:
  - `cd ~/workspace/code/high-level-robot-planner`
- Login-node policy:
  - Never run training on login nodes.
  - Use login nodes only for submission, monitoring, code sync, and lightweight ops.
  - Launch compute allocations/jobs for any actual training.
- Check queue:
  - `squeue --me`
- Monitor one job:
  - `squeue -j <JOBID> -o "%.18i %.30P %.20j %.8T %.10M %.9l %R"`
- Tail logs:
  - `tail -f <RUN_DIR>/<JOBID>.out`
  - `tail -f <RUN_DIR>/<JOBID>.err`
- Final status:
  - `sacct -j <JOBID> --format=JobID,State,ExitCode,Partition,Elapsed -n`

## Submit Workflow
- General form:
  - `python scripts/submit_job.py experiment=<experiment_name> [overrides...]`
- Dry-run generated sbatch only:
  - `python scripts/submit_job.py submit.dry_run=true experiment=<experiment_name>`
- Sweep support:
  - define `sweep.params` in the experiment config
  - submit all combinations:
    - `python scripts/submit_job.py experiment=<sweep_experiment>`
  - preview generated jobs:
    - `python scripts/submit_job.py submit.dry_run=true experiment=<sweep_experiment>`
- Local non-Slurm run path (when suitable):
  - run training scripts directly, e.g. `python scripts/2_train_stage1_lam.py ...`
  - this is common on workstation and sometimes in cluster interactive sessions

## Dual-Queue Strategy (Faster Start)
Submit the same run to both clusters:
- LRZ broad queue:
  - `cluster=lrz_x100` (partitions include H100/A100 pools)
- MCML broad queue:
  - `cluster=mcml_x100` (partitions include H100/A100 pools)

Typical sequence:
1. Submit LRZ job (`cluster=lrz_x100`)
2. Submit MCML job (`cluster=mcml_x100`)
3. Watch both job IDs in `squeue`
4. When one switches to `RUNNING`, cancel the other:
   - `scancel <other_job_id>`

## Container/Image Notes
- Current state: two separate containers are used.
  - Container A: latent action model (stage-1) and Stage-2 policy training.
  - Container B: LeRobot/stage-3 environment.
  - This is work in progress and may be unified later.
- Stage-1/2 image:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh`
- Stage-3 image:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh`
- If image import/build is needed:
  - Run Enroot import on a compute node (not login node if Enroot unavailable there).
  - Enroot import can OOM; request enough memory.

## Testing
- Run tests with `pytest` (user may call this "pi test").
- On workstation/local shells, run tests in `hlrp` env (e.g. `conda run -n hlrp pytest ...`).
- Prefer targeted tests for fast iteration before broader test suites.

## Run + Cache Paths
- Runs (timestamped):
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/<timestamp>_<experiment>`
- Cache root:
  - `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/cache`
- `submit_job.py` mounts repo/runs/cache into container automatically.

## W&B and Hugging Face Auth
- W&B:
  - Enable with `logging.use_wandb=true lerobot.wandb.enable=true`
  - In current setup, auth works from user credentials on cluster (e.g. `~/.netrc`) when accessible in runtime environment.
- Hugging Face:
  - `submit_job.py` looks for `~/.huggingface/token` and wires `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` for jobs.

## Known Stage 3 Installer Behavior
- `scripts/6_train_lerobot.py` editable policy install fallback order:
  1. `python -m pip install --no-deps -e ...`
  2. `uv pip install --python <current_python> --no-deps -e ...`
  3. `pip install --no-deps -e ...`
- This is to support different container layouts (with/without pip in active venv).

## Useful Stage 3 Smoke Command Pattern
For short stage-3 train-only smoke or verification runs, do not use the cluster default multi-hour limit for 50-200 step checks.

For stage-3 smoke runs with eval enabled, `00:15:00` is too short and `00:30:00` is still tight. Use at least `cluster.compute.time_limit=00:45:00`.

Example via submit script:
- `python scripts/submit_job.py experiment=stage3_cluster stage3_profile=action_scratch cluster=lrz_x100 cluster.compute.time_limit=00:45:00 experiment.name=stage3_hlrp_libero_smoke_wandb lerobot.steps=50 lerobot.batch_size=2 lerobot.eval.freq=10 lerobot.eval.batch_size=1 lerobot.eval.n_episodes=1 lerobot.log_freq=10 lerobot.save_freq=1000 logging.use_wandb=true lerobot.wandb.enable=true`

## Repo-local Skills

### Available skills
- `hlrp-docker-enroot-refresh`: Build, push, prune, import, and safely swap the HLRP stage-1/2 or stage-3 container using `containers/Dockerfile.stage12` or `containers/Dockerfile.stage3`, the bundled `lerobot/` Docker build context in this repo when needed, and the `felix_minzenmay/enroot/hlrp_stage12.sqsh` or `felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh` cluster paths. (file: `/mnt/data/workspace/code/high-level-robot-planner/.codex/skills/hlrp-docker-enroot-refresh/SKILL.md`)

### How to use repo-local skills
- If the user asks to rebuild, push, import, replace, or debug the cluster container workflow, open the skill and follow it.
