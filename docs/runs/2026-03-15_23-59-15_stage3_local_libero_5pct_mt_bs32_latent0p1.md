# Run Note: 2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1

## Meta

- Date: 2026-03-15
- Status: completed
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 04970c0557a41f7b3d76d15cabf04df7ed8f7883 (dirty: config/data/octo24*.yaml, containers/Dockerfile.unified, docs/runs/, scripts/submit_job.py)
- Worktree State: dirty
- Logical Cluster Target:
- Stage: stage3
- Script: `scripts/6_train_lerobot.py`
- Base Experiment: `stage3_local`
- Config Path: `config/experiment/runs/2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1.yaml`
- Experiment Name: `stage3_local` (launched before documented config was created; run dir uses default naming)
- Intended Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_23-59-15_stage3_local`
- Final Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_23-59-15_stage3_local`
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Ablation baseline for `2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`.
- Train Stage 3 multitask on only the 5% action-labeled Libero subset (84 episodes), discarding the 95% latent-only data.
- Same LAM checkpoint, same latent loss weight 0.1, but batch size 32 (profile default) instead of 64.
- Measures whether the extra 95% latent-only training data actually helps or hurts compared to multitask on the labeled subset alone.

## Config Delta Vs Default

- Override `stage3_dataset` from `libero` (full dataset) to `libero_5pct` (84 multitask-only episodes).
- Set `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`.
- Override `lerobot.policy.latent_loss_weight` from `1.0` to `0.1`.
- Override `lerobot.num_workers` from `8` to `4`.
- `lerobot.batch_size` stays at profile default `32` (vs 64 in the comparison run).
- All other settings (model, scheduler, optimizer, eval, steps=100k) remain at `_stage3_base` / `multitask_scratch` defaults.

## Upstream Artifacts / Checkpoints

- Type: `stage1 checkpoint`
  Source Run: `2026-03-14_19-27-26_stage1_local`
  Path: `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`
  Notes: Libero-only LAM checkpoint used for latent supervision, same as in the comparison run.

## Launch Command

```bash
conda run -n lerobot python scripts/6_train_lerobot.py \
  experiment=stage3_local \
  stage3_dataset=libero_5pct \
  artifacts.lam_checkpoint_path=/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt \
  lerobot.batch_size=32 \
  lerobot.num_workers=4 \
  lerobot.policy.latent_loss_weight=0.1 \
  logging.use_wandb=true \
  lerobot.wandb.enable=true
```

tmux session: `stage3_5pct`

## Results / Findings

- Launched locally on `tueilsy-st-022`; `unified.log` records the LeRobot train command launch at `2026-03-15 23:59:17 CET`.
- Stage 3 training completed successfully at `2026-03-16 13:45:31 CET` according to `unified.log`, for a runtime of about `13h46m` (`49565s` in the W&B summary).
- Saved checkpoints at `020000`, `040000`, `060000`, `080000`, and `100000`, with `lerobot/checkpoints/last -> 100000`.
- Training stayed stable throughout: logged loss fell from `0.331` at step `200` to about `0.015` at `20k`, `0.009` at `40k`, `0.005` at `60k`, `0.003` at `80k`, and finished with W&B summary metrics `train/loss=0.001411`, `action_loss=0.001144`, `latent_loss=0.002655`, `grad_norm=0.0388`.
- Final training summary at `100k`: `samples=3.2M`, `episodes=20049.23`, `epochs=238.68`, `lr=2.5e-05`, `action_supervised_fraction=1.0`, `latent_supervised_fraction=0.96875`.
- Final in-training `libero_10` evaluation at `100k` reached `15%` success with `avg_sum_reward=0.15`.
- The run is still weak on held-out rollout quality from the training-time eval alone: only `15/100` successes on `libero_10`, so a full four-suite rollout remains necessary before drawing broader conclusions.

## Rollout Evaluation

### 2026-03-16 local rollout results

- Rollout run: `2026-03-16_14-25-59_stage3_rollout_local_from_2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1`
- Checkpoint: `lerobot/checkpoints/100000/pretrained_model` (step 100k)
- Launched locally at `2026-03-16 14:27:48 CET` and completed at `2026-03-16 15:37:04 CET` according to `unified.log` (`~69m` wall time).
- Overall rollout result: `25.0%` success over `400` episodes (`40` tasks x `10` episodes), with `avg_sum_reward=0.25` and `avg_max_reward=0.25`.
- Group-level success split: `libero_spatial 13%`, `libero_object 38%`, `libero_goal 41%`, `libero_10 8%`.
- Task outcomes were weak and uneven: `22` tasks had zero successes, `15` were partial-success tasks, and only `3` achieved `10/10` success.
- `libero_goal` and `libero_object` carried most of the performance; `libero_spatial` and especially `libero_10` remained poor despite the clean training curve.
- Compared with the balanced `95% latent / 5% multitask` run, this 5%-only rollout underperformed substantially (`25.0%` overall here vs `35.8%` there), which supports the value of the latent-only portion for final control quality.
- Outputs include `eval_info.json`, `400` rollout videos under `videos/`, and the launcher log in this rollout run directory.

```bash
conda run -n lerobot python scripts/7_rollout_lerobot.py \
  experiment=stage3_rollout_local \
  experiment.name=2026-03-16_14-25-59_stage3_rollout_local_from_2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1 \
  logging.runs_dir=/mnt/data/workspace/runs_root/runs/2026-03-16_14-25-59_stage3_rollout_local_from_2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1 \
  lerobot_eval.policy_path=/mnt/data/workspace/runs_root/runs/2026-03-15_23-59-15_stage3_local/lerobot/checkpoints/100000/pretrained_model
```

| Scope | Success % | Successes | Episodes | Avg Sum Reward | Avg Max Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 25.0 | 100 | 400 | 0.25 | 0.25 |
| libero_spatial | 13.0 | 13 | 100 | 0.13 | 0.13 |
| libero_object | 38.0 | 38 | 100 | 0.38 | 0.38 |
| libero_goal | 41.0 | 41 | 100 | 0.41 | 0.41 |
| libero_10 | 8.0 | 8 | 100 | 0.08 | 0.08 |

| Task Outcome Bucket | Count |
| --- | ---: |
| 0/10 success tasks | 22 |
| Partial-success tasks | 15 |
| 10/10 success tasks | 3 |
