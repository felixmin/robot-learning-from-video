# Run Note: 2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local

## Meta

- Date: 2026-03-10
- Status: completed
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 9ba17e837fd5c5102fb6967329f6f9fd66a2452e
- Worktree State: dirty (`.codex/skills/start-run/SKILL.md`, `docs/runs/TEMPLATE.md`, `lerobot`)
- Logical Cluster Target:
- Stage: rollout
- Script: `scripts/7_rollout_lerobot.py`
- Base Experiment: `stage3_rollout_local`
- Config Path: `config/experiment/runs/2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local.yaml`
- Experiment Name: `2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local`
- Intended Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local`
- Final Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local`
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Run dedicated local Stage 3 rollouts for the final checkpoint from `2026-03-09_23-54-36_stage3_local`.
- The source Stage 3 training run used the `libero_5pct` dataset subset rather than the full Libero training data.
- Evaluate the exported LeRobot policy with `scripts/7_rollout_lerobot.py` instead of relying only on training-time evals.

## Config Delta Vs Default

- Set `experiment.name` to the dated documented rollout stem.
- Set `experiment.description` for this source run.
- Set `logging.runs_dir` to `${logging.root_dir}/runs/${experiment.name}` to avoid duplicate timestamps in the output path.
- Set `lerobot_eval.policy_path` to `/mnt/data/workspace/runs_root/runs/2026-03-09_23-54-36_stage3_local/lerobot/checkpoints/last/pretrained_model`.

## Upstream Artifacts / Checkpoints

- Type: `policy path`
  Source Run: `2026-03-09_23-54-36_stage3_local`
  Path: `/mnt/data/workspace/runs_root/runs/2026-03-09_23-54-36_stage3_local/lerobot/checkpoints/last/pretrained_model`
  Notes: Final saved Stage 3 policy used as rollout input; the source training run used the `libero_5pct` dataset subset.
- Type: `stage1 checkpoint`
  Source Run: `2026-02-24_10-41-51_laq_oxe_local`
  Path: `/mnt/data/workspace/runs_root/runs/moved_from_repo_dir/runs/2026-02-24_10-41-51_laq_oxe_local/checkpoints/last.ckpt`
  Notes: Resolved from the source Stage 3 run's saved Hydra config and launch command as the `policy.lam_checkpoint_path` used for latent supervision.

## Launch Command

```bash
conda run -n lerobot python scripts/7_rollout_lerobot.py experiment=runs/2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local
```

## Results / Findings

- Launched locally on 2026-03-10 at 14:18:32 CET and completed at 2026-03-10 15:24:09 CET.
- Overall rollout result: `31.0%` success over `400` episodes (`40` tasks x `10` episodes), with `avg_sum_reward=0.31` and `avg_max_reward=0.31`.
- Group-level success split: `libero_spatial 22%`, `libero_object 23%`, `libero_goal 57%`, `libero_10 22%`.
- Task outcomes were highly uneven: `20` tasks had zero successes, `15` were partial, and `5` achieved `10/10` success.
- Strongest group was `libero_goal`; the other three groups clustered near `22-23%`, indicating weak transfer outside that group.
- Outputs include `eval_info.json`, `400` rollout videos under `videos/`, and the launcher log at `/mnt/data/workspace/runs_root/runs/2026-03-10_14-15-37_stage3_rollout_local_from_2026-03-09_23-54-36_stage3_local/unified.log`.

| Scope | Success % | Successes | Episodes | Avg Sum Reward | Avg Max Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 31.0 | 124 | 400 | 0.31 | 0.31 |
| libero_spatial | 22.0 | 22 | 100 | 0.22 | 0.22 |
| libero_object | 23.0 | 23 | 100 | 0.23 | 0.23 |
| libero_goal | 57.0 | 57 | 100 | 0.57 | 0.57 |
| libero_10 | 22.0 | 22 | 100 | 0.22 | 0.22 |

| Task Outcome Bucket | Count |
| --- | ---: |
| 0/10 success tasks | 20 |
| Partial-success tasks | 15 |
| 10/10 success tasks | 5 |
