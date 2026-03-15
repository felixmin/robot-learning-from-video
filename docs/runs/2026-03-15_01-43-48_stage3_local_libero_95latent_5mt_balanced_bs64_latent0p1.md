# Run Note: 2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1

## Meta

- Date: 2026-03-15
- Status: running
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
- Training is still active. As of `2026-03-15 17:20:56 CET`, the run has completed checkpoint saves through `80k` steps and is currently inside the `80k` evaluation pass.
- Checkpoint schedule so far: `20k`, `40k`, `60k`, and `80k`, with `lerobot/checkpoints/last -> 080000`.
- The last saved training-state snapshot records `step=80000`; the latest scheduler state reports learning rate `1.1738457630498262e-04`.
- Training loss improved from `0.325` at step `200` to about `0.004` by the `80k` checkpoint interval.
- Evaluation trend improved from `0%` success at `20k` to `7%` overall success on `libero_10` at `60k` (`100` episodes, `avg_sum_reward=0.07`, `avg_max_reward=0.07`).
- The `80k` evaluation is not finished yet. The current log tail shows early task batches with provisional running success rates of `0%`, `0%`, and `10%`, so no final `80k` aggregate should be recorded yet.
- An earlier local Stage 3 attempt with the same Stage 1 checkpoint exists at `2026-03-15_01-20-13_stage3_local_libero_95latent_5mt_balanced_bs64_lam50k`; this documented run is the follow-up `latent_loss_weight=0.1` variant.
