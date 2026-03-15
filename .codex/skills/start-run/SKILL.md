---
name: start-run
description: Start a local or cluster HLRP run across Stage 1, Stage 2, Stage 3, or rollout. Default to a documented major run that creates a dated config in `config/experiment/runs/` and a matching note in `docs/runs/`. Only skip that documentation flow when the user explicitly says `test` or `debug`. Post-training rollouts launched with `scripts/7_rollout_lerobot.py` are treated as evaluation attached to the source Stage 3 run note rather than as a new documented run. Use direct stage scripts for local runs and `scripts/submit_job.py` for cluster runs. For the logical `lrz cluster`, dual-submit to `lrz_x100` and `mcml_x100`, then cancel the losing job with the 1-hour predicted-start rule.
---

# Start Run

Use when the user wants to start or resume a training or rollout run.

## First checks

Always check:
- `pwd`
- `hostname`
- whether you are on the workstation or already on the cluster

Obey an explicit user choice of `local` or `cluster`.

If the user says `cluster` without naming a cluster, assume the logical `lrz cluster` flow:
- submit one job with `cluster=lrz_x100`
- submit one job with `cluster=mcml_x100`
- monitor both and cancel the loser

## Run class

Default to a documented major run.

Only treat the run as lightweight and undocumented when the user explicitly says `test` or `debug`.

Rules:
- Major run: create a dated experiment config and a matching docs note before launch.
- Lightweight run: do not create a new experiment config and do not create a docs note.
- Post-training rollout for an existing Stage 3 run: do not create a new docs note; append the rollout launch and results to the source Stage 3 run note.
- Lightweight runs should use the closest existing stage experiment plus CLI overrides.

## Stage mapping

Pick the base experiment and entrypoint from this table:

- Stage 1 local:
  - base experiment: `stage1_local`
  - entrypoint: `scripts/2_train_stage1_lam.py`
  - local env: `hlrp`
- Stage 1 cluster:
  - base experiment: `stage1_cluster`
  - submit script: `2_train_stage1_lam`
- Stage 2 local:
  - base experiment: `stage2_local`
  - entrypoint: `scripts/4_train_stage2_policy.py`
  - local env: `hlrp`
- Stage 2 cluster:
  - base experiment: `stage2_cluster`
  - submit script: `4_train_stage2_policy`
- Stage 3 local:
  - base experiment: `stage3_local`
  - entrypoint: `scripts/6_train_lerobot.py`
  - local env: `lerobot`
- Stage 3 cluster:
  - base experiment: `stage3_cluster`
  - submit script: `6_train_lerobot`
- Rollout local:
  - base experiment: `stage3_rollout_local`
  - entrypoint: `scripts/7_rollout_lerobot.py`
  - local env: `lerobot`
- Rollout cluster:
  - base experiment: `stage3_rollout_cluster`
  - submit script: `7_rollout_lerobot`

Rollout-specific contract:
- `lerobot_eval.policy_path` must point to the exported `pretrained_model` directory inside a saved checkpoint.
- Use paths such as `.../lerobot/checkpoints/last/pretrained_model` or `.../lerobot/checkpoints/100000/pretrained_model`.
- Do not point `lerobot_eval.policy_path` at the checkpoint root such as `.../checkpoints/last` without the trailing `pretrained_model`.
- When the rollout evaluates a completed or existing Stage 3 training run, treat it as evaluation of that training run, not as a separate experiment lineage.

## Naming contract

Use an expressive stem:

`YYYY-MM-DD_HH-MM-SS_<what_this_run_is>`

Examples:
- `2026-03-10_14-20-00_stage1_local_5pct_real_action`
- `2026-03-10_14-30-00_stage3_cluster_action_scratch_libero`

Rules:
- For major runs, `experiment.name` must equal the full stem.
- The config path must be `config/experiment/runs/<stem>.yaml`.
- The docs path must be `docs/runs/<stem>.md`.
- Keep the stem descriptive enough to show the stage, local or cluster mode, and the key ablation or purpose.

Important:
- The repo default `logging.runs_dir` is `${logging.root_dir}/runs/${now:%Y-%m-%d_%H-%M-%S}_${experiment.name}`.
- That duplicates the timestamp for major runs because `experiment.name` already contains the timestamp.
- For every major run, explicitly set `logging.runs_dir` so the run directory is based on `experiment.name`, not on a second `now`.

## Major local-run workflow

1. Choose the closest base experiment from the stage table above.
2. Create `config/experiment/runs/<stem>.yaml` by copying the matching base experiment config and editing only what differs.
3. In that config:
   - set `experiment.name: <stem>`
   - update `experiment.description`
   - keep the config minimal and fail-fast
   - set `logging.runs_dir: ${logging.root_dir}/runs/${experiment.name}`
   - keep the base config's `defaults` entries directly in the run config; do not compose the run config via `defaults: - /experiment: <base_experiment>` from inside `config/experiment/runs`, because that can recurse through the experiment group
4. Create `docs/runs/<stem>.md` from `docs/runs/TEMPLATE.md`.
5. Fill in the metadata before launch:
   - date
   - mode
   - host
   - current code commit reference
   - stage
   - script
   - base experiment
   - config path
   - experiment name
   - intended run dir
   - purpose
   - config delta versus default
   - upstream artifacts and checkpoints used
6. Launch the stage script directly, not through `scripts/submit_job.py`.

Local command pattern:

```bash
conda run -n <env> python <script> experiment=runs/<stem> [extra overrides]
```

Use:
- `hlrp` for Stage 1 and Stage 2
- `lerobot` for Stage 3 and rollout

Durability rule for long local runs:
- Prefer launching from `tmux`, not from an ad hoc background job.
- If a long-lived session already exists, create a new window in that session.
- Otherwise create a dedicated detached session before starting the command.

## Major cluster-run workflow

1. Choose the closest cluster base experiment from the stage table above.
2. Create `config/experiment/runs/<stem>.yaml` by copying the matching base experiment config and editing only what differs.
3. In that config:
   - set `experiment.name: <stem>`
   - update `experiment.description`
   - set `logging.runs_dir: ${logging.root_dir}/runs/${experiment.name}`
4. Create `docs/runs/<stem>.md` from `docs/runs/TEMPLATE.md`.
5. Fill in the metadata before launch, including the logical target cluster.
   Record the current code commit reference from `git rev-parse HEAD`. If the worktree is dirty, note that explicitly in the docs entry.
   Also record every reused upstream artifact or checkpoint with both:
   - the documented run stem it came from, when known
   - the explicit resolved path used at launch time
6. Submit through `scripts/submit_job.py`.

When the user says `cluster`, default to the logical `lrz cluster` flow:

- submit one job with `cluster=lrz_x100`
- submit one job with `cluster=mcml_x100`
- keep the same `experiment.name`
- override the run dirs so the two queued jobs do not write into the same folder

Use queue-specific run directories:
- LRZ job: `${logging.root_dir}/runs/${experiment.name}_lrz`
- MCML job: `${logging.root_dir}/runs/${experiment.name}_mcml`

Record both job IDs and both run dirs in the docs note.

## LRZ dual-submit policy

This dual-submit behavior is specific to the logical `lrz cluster`.

Workflow:
1. Submit the LRZ broad-queue job.
2. Submit the MCML broad-queue job.
3. Check both jobs with `squeue` and `squeue --start`.
4. If one job reaches `RUNNING`, cancel the other immediately.
5. If both predicted start times are more than 1 hour away, cancel the one with the later predicted start time and keep the earlier one.
6. If both predicted start times are within 1 hour, wait for a start event, but never wait for 1 hour or longer.
7. Cap the waiting window below 1 hour. Use a concrete cap such as 55 minutes.
8. If neither job starts before that cap, stop waiting, report both statuses, and tell the user which job currently looks better.

If `squeue --start` does not provide a prediction:
- fall back to queue state and current priority signals
- report the uncertainty explicitly

## Lightweight local-run workflow

Do not create `config/experiment/runs/...` or `docs/runs/...`.

Use the closest base experiment and direct CLI overrides.

Pattern:

```bash
conda run -n <env> python <script> experiment=<base_experiment> experiment.name=<expressive_name> [debug overrides]
```

Rules:
- still make `experiment.name` expressive
- prefer CLI overrides over new config files
- use this only for explicit `test` or `debug` requests

## Lightweight cluster-run workflow

Do not create `config/experiment/runs/...` or `docs/runs/...`.

Use the closest existing cluster experiment and submit with CLI overrides:

```bash
python scripts/submit_job.py experiment=<base_experiment> submit.script=<submit_script> experiment.name=<expressive_name> [overrides]
```

If the user says `cluster` without naming a cluster, use the LRZ dual-submit flow above.

## Documentation rules for major runs

Before launch, the docs note must state:
- whether the run is local or cluster
- the current code commit reference used to launch the run
- the canonical config name
- the base experiment it was derived from
- the exact config delta from the default base experiment
- every reused upstream artifact or checkpoint, including both the source run stem and the explicit path
- the intended run directory or directories
- the purpose of the run

After launch, update the same note with:
- job IDs
- the winning cluster submission, if applicable
- final run directory

## Artifact and checkpoint provenance

Document reused artifacts explicitly.

Rules:
- Stage 1 usually has no upstream checkpoint. State that clearly when none is used.
- Stage 2, Stage 3, and rollout runs usually depend on an earlier pipeline artifact or checkpoint. Record that provenance before launch.
- For each reused artifact or checkpoint, include both:
  - the source documented run stem, for example `2026-03-10_14-20-00_stage1_local_5pct_real_action`
  - the explicit path used at launch time
- If the source run is not documented, still record the explicit path and say that the source run stem is unknown.
- If multiple upstream artifacts are used, list all of them separately.

## Resume runs

If the user is resuming a run:
- prefer reusing the documented config when one exists
- keep `experiment.name` aligned with the documented stem unless there is a deliberate new run
- set resume-specific CLI overrides explicitly
- ensure any step-based limits still allow additional work

If the user asks to start rollouts from an existing Stage 3 training run:
- do not create a new `docs/runs/<stem>.md`
- append rollout planning, launch details, and eventual results to the source training run note
- keep the evaluation visually separate inside that note under a new bottom section such as `## Rollout Evaluation` or an additional dated rollout subsection
- point `lerobot_eval.policy_path` at the source run's saved `pretrained_model` directory
- document the source checkpoint path, rollout command, run directory, and monitoring details in that existing note
- prefer direct CLI overrides for the rollout command
- only create a dedicated rollout config when the rollout setup is complex enough that reproducing it from CLI overrides would be error-prone
- if a dedicated rollout config is created for reproducibility, do not create a separate rollout markdown note for it
- for local rollout launches, record the `tmux` session and window used for monitoring

## Rollout documentation contract

For post-training rollouts tied to an existing Stage 3 run:
- the canonical markdown note remains the source Stage 3 training note
- the rollout should be documented as evaluation attached to that run, because `scripts/7_rollout_lerobot.py` does not produce new weights
- include at minimum:
  - rollout date
  - local or cluster mode
  - rollout command
  - rollout run directory
  - resolved `lerobot_eval.policy_path`
  - final metrics and qualitative findings when available
- if there are multiple rollout passes for the same training run, append them chronologically within the same note rather than creating sibling rollout notes

## Output standard

When you finish, give the user the exact launch command or commands, the resolved run directory or directories, and for major runs the exact config path plus docs path.
