---
name: evaluate-run
description: Evaluate an HLRP run from a flexible reference such as a job id, run directory, documented run stem, docs note, config path, or the phrase `check last run`. For documented runs, update the matching file in `docs/runs/` once the run has finished. For lightweight or still-active runs, return a normal evaluation summary without writing docs.
---

# Evaluate Run

Use when the user asks to inspect, evaluate, analyze, or summarize a run.

## First checks

Always check:
- `pwd`
- `hostname`
- whether you are on the workstation or already on the cluster

If the run reference is clearly local, stay local.

If the run reference is clearly cluster-side, or if the run is missing locally, inspect the cluster via `ssh ai`.

## Accepted run references

Accept all of these:
- a Slurm job ID
- a run directory path
- a docs note path in `docs/runs/`
- a config path in `config/experiment/runs/`
- a documented stem such as `2026-03-10_14-20-00_stage1_local_5pct_real_action`
- the phrase `check last run`

## `check last run`

Resolve `check last run` from the current system configuration:

1. Prefer `config/user_config/local.yaml` when it exists.
2. Otherwise fall back to `config/user_config/local.yaml.example`.
3. Otherwise fall back to `config/config.yaml`.

Use those files to determine:
- `logging.root_dir`
- `logging.runs_dir`

Resolution rules:
- if there is a fixed `logging.runs_dir`, inspect its parent directory for sibling run folders
- otherwise inspect `<logging.root_dir>/runs`
- otherwise inspect the repo-local default `./runs`

Choose the newest run directory by modification time.

Important:
- explicitly mention that the last run may still be active
- if the local path does not exist and the request likely refers to the cluster, inspect the cluster runs root via `ssh ai`

## Documented vs lightweight runs

Treat a run as documented when either of these exists:
- `docs/runs/<stem>.md`
- `config/experiment/runs/<stem>.yaml`

Otherwise treat it as lightweight or ad hoc.

## Resolution order

Use this order:
1. direct path supplied by the user
2. docs note stem or config stem
3. Slurm job ID lookup
4. `check last run`

For cluster job IDs, follow the same locating pattern used by the existing cluster-analyzer workflow:
- inspect `squeue`
- inspect `sacct`
- locate `<jobid>.out` and `<jobid>.err` under the cluster runs root
- derive the run directory from the log path

## Evaluation workflow

Always determine:
- run status: running, pending, completed, failed, canceled, unknown
- run directory
- config snapshot or source config
- stage and mode if you can infer them

Then inspect the main artifacts that exist:
- `unified.log`
- `.out` and `.err`
- `.hydra/`
- checkpoints
- visualizations
- profiler outputs
- W&B or LeRobot outputs when present

Return a normal evaluation with the most relevant signals:
- training loss trend
- validation trend
- success or rollout metrics
- signs of overfitting
- instability, crashes, or data issues
- whether the run is still progressing or has plateaued

## Stage-specific expectations

For Stage 1 and Stage 2:
- focus on train and validation losses
- look for qualitative validation outputs when present
- flag divergence, collapse, or overfitting risks

For Stage 3 train runs:
- inspect LeRobot training outputs under the run directory
- summarize train and eval behavior
- call out task coverage and any obvious evaluation weaknesses

For rollout runs:
- focus on rollout success, task-wise behavior, and failure patterns
- use aggregate success and qualitative artifacts when available
- for completed documented rollout runs, add markdown tables to the docs note for overall and per-group metrics, plus a compact task-outcome bucket table when those counts are available

## Active runs

If the run is still active:
- return a live summary of what is happening now
- do not write to `docs/runs/`
- mark the conclusions as provisional

## Docs update rules for documented runs

Only update `docs/runs/<stem>.md` when:
- the run is documented
- the run is no longer active
- you have a clear resolved match

When updating the docs note:
- keep the existing metadata intact unless it is wrong
- fill in final status
- add the resolved run directory
- add concise factual findings under `Results / Findings`
- for rollout runs, include the rollout metrics tables from the template when the underlying artifacts support them
- include enough detail to support later ablations or experiment writeups

Do not write docs when:
- the run is lightweight or undocumented
- the run is still active
- the reference is ambiguous

## Output standard

Always return:
- the resolved run reference
- the run status
- the run directory
- whether the run is documented
- the main findings

If you updated a docs note, also return its path.
