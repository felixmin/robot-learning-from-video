# Run Note: 2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask

## Meta

- Date: 2026-03-15
- Status: planned
- Mode: cluster
- Host: tueilsy-st-022
- Code Commit: 509ed113608417477fb94574b951f6cdf0fd56fa
- Worktree State: dirty; local skill docs and in-progress run/config notes are uncommitted during launch.
- Logical Cluster Target: lrz cluster
- Stage: stage1
- Script: scripts/2_train_stage1_lam.py
- Base Experiment: stage1_cluster
- Config Path: config/experiment/runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask.yaml
- Experiment Name: 2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask
- Intended Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask_{lrz|mcml}
- Final Run Dir:
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Produce a minimally trained but valid Stage 1 LAM checkpoint for later Stage 3 multitask Libero runs that require `artifacts.lam_checkpoint_path`.

## Config Delta Vs Default

- `experiment.name` set to the documented dated run stem.
- `experiment.description` narrowed to a checkpoint-bootstrap purpose.
- `logging.runs_dir` pinned to `${logging.root_dir}/runs/${experiment.name}` for documented-run naming.
- `logging.tags` narrowed to a checkpoint-bootstrap run.
- `training.max_steps=50` to keep the run short.
- `training.num_sanity_val_steps=0` to avoid startup-only sanity validation overhead.

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: none
  Path: none
  Notes: This run is the source checkpoint producer for a later Stage 3 multitask training run.

## Launch Command

```bash
# LRZ queue
python scripts/submit_job.py \
  experiment=runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask \
  cluster=lrz_x100 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask_lrz

# MCML queue
python scripts/submit_job.py \
  experiment=runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask \
  cluster=mcml_x100 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_18-24-58_stage1_cluster_tiny_lam_ckpt_for_stage3_multitask_mcml
```

## Results / Findings

- Pending.
