# Run Note: 2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1

## Meta

- Date: 2026-03-15
- Status: completed
- Mode: local
- Host: tueilsy-st-022
- Code Commit: 7038b673a6b9752420383941de425c3f1adb0848 (inferred from local `git reflog`; HEAD remained on this commit through the 2026-03-15 01:45 CET launch window)
- Worktree State: unknown (LeRobot W&B metadata did not preserve dirty/clean state)
- Logical Cluster Target:
- Stage: stage3
- Script: `scripts/6_train_lerobot.py`
- Base Experiment: `stage3_local`
- Config Path: `config/experiment/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1.yaml`
- Experiment Name: `2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`
- Intended Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`
- Final Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Train Stage 3 locally on the balanced `95% latent / 5% multitask` Libero mix using a Libero-only Stage 1 checkpoint for latent supervision.
- Continue the Libero-specialization workflow by taking the Stage 1 checkpoint from `2026-03-14_19-27-26_stage1_local` and applying it in local SmolVLA shared Stage 3 training.
- Reduce the latent-loss contribution to `0.1` while keeping the multitask-scratch profile and batch size `64`.

## Config Delta Vs Default

- Override `stage3_dataset` from the default full Libero Stage 3 dataset to `libero_5pct_latent_rest_balanced`.
- Set `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`.
- Override `lerobot.batch_size` from the current base-profile default to `64`, matching the historical run.
- Override `lerobot.policy.latent_loss_weight` from `1.0` to `0.1`.
- Override `lerobot.num_workers` from `8` to `4`.
- Set `experiment.name` to the dated run stem used by the local run directory.
- Set `logging.runs_dir` to `${logging.root_dir}/runs/${experiment.name}` so the documented config matches the historical run path.
- Update `experiment.description` for the Libero balanced-mix and latent-weight ablation used here.

## Upstream Artifacts / Checkpoints

- Type: `stage1 checkpoint`
  Source Run: `2026-03-14_19-27-26_stage1_local`
  Path: `/mnt/data/workspace/runs_root/runs/2026-03-14_19-27-26_stage1_local/checkpoints/last.ckpt`
  Notes: Libero-only LAM checkpoint used for latent supervision in the multitask Stage 3 run.

## Launch Command

```bash
conda run -n lerobot python scripts/6_train_lerobot.py experiment=runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1
```

## Results / Findings

- Launched locally on `tueilsy-st-022` at `2026-03-15 01:45:34 CET` according to the LeRobot W&B metadata.
- W&B run path: `felixmin/lerobot/66px0n4q`.
- The run uses `hlrp/libero_5pct_latent_rest_balanced` with `1693` episodes and `273465` frames, batch size `64`, and local RTX 5090 training.
- Stage 3 training completed at `2026-03-15 21:32:56 CET` according to `unified.log`, for a runtime of about `19h47m`.
- Saved checkpoints at `020000`, `040000`, `060000`, `080000`, and `100000`, with `lerobot/checkpoints/last -> 100000`.
- The final training-state snapshot records `step=100000`; the final scheduler state reports learning rate `2.5e-05`.
- Training loss improved from `0.325` at step `200` to `0.002716` at step `100000`.
- Final train-side summary at `100k`: `action_loss=0.001053`, `latent_loss=0.016724`, `grad_norm=0.0373`, `action_supervised_fraction=0.4375`, `latent_supervised_fraction=0.890625`.
- In-training `libero_10` evaluation improved monotonically across checkpoints:
  - `20k`: `0%` success, `avg_sum_reward=0.00`
  - `40k`: `6%` success, `avg_sum_reward=0.06`
  - `60k`: `7%` success, `avg_sum_reward=0.07`
  - `80k`: `13%` success, `avg_sum_reward=0.13`
  - `100k`: `20%` success, `avg_sum_reward=0.20`
- The run appears stable throughout training: loss decays cleanly, no crash markers appear in the local logs, and the final checkpoint/eval pair completed successfully.
- An earlier local Stage 3 attempt with the same Stage 1 checkpoint exists at `2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k`; this documented run is the follow-up `latent_loss_weight=0.1` variant.

## Rollout Evaluation

### 2026-03-15 local rollout results

- Rollout run: `2026-03-15_22-09-51_stage3_rollout_local_from_2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`
- Checkpoint: `lerobot/checkpoints/last/pretrained_model` (step 100k)
- Launched locally at `2026-03-15 22:12:14 CET` and completed at `2026-03-15 23:18:18 CET` according to `unified.log` (`~66m` wall time).
- Overall rollout result: `35.8%` success over `400` episodes (`40` tasks x `10` episodes), with `avg_sum_reward=0.3575` and `avg_max_reward=0.3575`.
- Group-level success split: `libero_spatial 37%`, `libero_object 40%`, `libero_goal 51%`, `libero_10 15%`.
- Task outcomes were still uneven: `17` tasks had zero successes, `17` were partial-success tasks, and `6` achieved `10/10` success.
- `libero_goal` remained the strongest suite, while `libero_10` was clearly the weakest despite the stronger balanced-mix performance on the other three suites.
- Outputs include `eval_info.json`, `400` rollout videos under `videos/`, and the launcher log in this rollout run directory.

| Scope | Success % | Successes | Episodes | Avg Sum Reward | Avg Max Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 35.8 | 143 | 400 | 0.3575 | 0.3575 |
| libero_spatial | 37.0 | 37 | 100 | 0.37 | 0.37 |
| libero_object | 40.0 | 40 | 100 | 0.40 | 0.40 |
| libero_goal | 51.0 | 51 | 100 | 0.51 | 0.51 |
| libero_10 | 15.0 | 15 | 100 | 0.15 | 0.15 |

| Task Outcome Bucket | Count |
| --- | ---: |
| 0/10 success tasks | 17 |
| Partial-success tasks | 17 |
| 10/10 success tasks | 6 |
