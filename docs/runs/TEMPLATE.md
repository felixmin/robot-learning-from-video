# Run Note: <stem>

## Meta

- Date:  # Launch date or planned launch date in `YYYY-MM-DD`.
- Status: planned  # Current state such as `planned`, `running`, `completed`, `failed`, or `canceled`.
- Mode: local  # High-level execution mode: `local` or `cluster`.
- Host:  # Concrete machine used to launch or inspect the run, for example the workstation hostname or cluster login host.
- Code Commit:  # Exact git commit hash used when the run was launched.
- Worktree State:  # Usually `clean` or `dirty`; note important uncommitted changes if dirty.
- Logical Cluster Target:  # User-facing cluster target or routing policy, for example `lrz cluster`; leave blank for pure local runs.
- Stage:  # Pipeline stage such as `stage1`, `stage2`, `stage3`, or `rollout`.
- Script:  # Exact entrypoint script, for example `scripts/2_train_stage1_lam.py`.
- Base Experiment:  # Existing base experiment this run was derived from, for example `stage2_local`.
- Config Path:  # Canonical config path for documented runs, usually `config/experiment/runs/<stem>.yaml`.
- Experiment Name:  # Resolved `experiment.name` used by Hydra and run naming.
- Intended Run Dir:  # Output directory planned at launch time from config or submission logic.
- Final Run Dir:  # Output directory that actually corresponds to the finished or winning run once resolved.
- LRZ Job ID:  # Slurm job id for the LRZ submission when applicable.
- MCML Job ID:  # Slurm job id for the MCML submission when applicable.

## Purpose

-

## Config Delta Vs Default

-

## Upstream Artifacts / Checkpoints

- Type:  # For example `stage1 checkpoint`, `stage2 artifact`, `stage3 checkpoint`, `policy path`, or `rollout input`.
  Source Run:  # Documented run stem such as `2026-03-10_14-20-00_stage1_local_5pct_real_action`; write `none` or `unknown` when applicable.
  Path:  # Explicit filesystem path used for this run, including rollout policy or checkpoint paths.
  Notes:  # Optional short note on how this artifact is used in the current run, for example train init, finetune source, or rollout policy input.

## Launch Command

```bash
# Fill in the exact command that launched the run.
```

## Results / Findings

- Pending.

For completed rollout runs, add rollout metrics tables such as:

| Scope | Success % | Successes | Episodes | Avg Sum Reward | Avg Max Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall |  |  |  |  |  |
| <task_group> |  |  |  |  |  |
| <task_group> |  |  |  |  |  |

| Task Outcome Bucket | Count |
| --- | ---: |
| 0/10 success tasks |  |
| Partial-success tasks |  |
| 10/10 success tasks |  |
