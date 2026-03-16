# Run Note: 2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step

## Meta

- Date: 2026-03-16
- Status: running
- Mode: local
- Host: tueilsy-st-022
- Code Commit: cd404ec7d56d159222ab2b31254e852b3544fb03
- Worktree State: dirty (`docs/runs/2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1.md` and `docs/runs/2026-03-15_23-59-15_stage3_local_libero_5pct_mt_bs32_latent0p1.md` modified before launch)
- Logical Cluster Target:
- Stage: stage3
- Script: `scripts/6_train_lerobot.py`
- Base Experiment: `stage3_local`
- Config Path: `config/experiment/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step.yaml`
- Experiment Name: `2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step`
- Intended Run Dir: `/mnt/data/workspace/runs_root/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step`
- Final Run Dir: /mnt/data/workspace/runs_root/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step
- LRZ Job ID:
- MCML Job ID:

## Purpose

- Re-run the local balanced `95% latent / 5% multitask` Stage 3 Libero setup with the same batch size `64` and latent loss weight `0.1`, but replace the prior Libero-specialized Stage 1 checkpoint with a fresh 3-step Stage 1 ablation checkpoint.
- Measure how much Stage 3 depends on a meaningful Stage 1 latent model versus a near-random Stage 1 checkpoint under the same dataset mix and training configuration.

## Config Delta Vs Default

- Override `stage3_dataset` from the default full Libero Stage 3 dataset to `libero_5pct_latent_rest_balanced`.
- Set `artifacts.lam_checkpoint_path` to `/mnt/data/workspace/runs_root/runs/2026-03-16_17-06-13_stage1_local_random_init/checkpoints/last.ckpt`.
- Override `lerobot.batch_size` from the current base-profile default to `64`.
- Override `lerobot.policy.latent_loss_weight` from `1.0` to `0.1`.
- Override `lerobot.num_workers` from `8` to `4`.
- Set `experiment.name` to the dated run stem used by the local run directory.
- Set `logging.runs_dir` to `${logging.root_dir}/runs/${experiment.name}` so the run dir is stable and does not inherit the historical source config path.

## Upstream Artifacts / Checkpoints

- Type: `stage1 checkpoint`
  Source Run: `undocumented local run: stage1_local_random_init`
  Path: `/mnt/data/workspace/runs_root/runs/2026-03-16_17-06-13_stage1_local_random_init/checkpoints/last.ckpt`
  Notes: 3-step Stage 1 ablation checkpoint used as the latent-supervision source for this Stage 3 run.

## Launch Command

```bash
conda run -n lerobot python scripts/6_train_lerobot.py experiment=runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step
```

## Results / Findings

- Launched locally at `2026-03-16 17:31:28 CET` according to `unified.log`.
- Running in tmux session `stage3_randlam3`.
- Run directory resolved as `/mnt/data/workspace/runs_root/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step`.
- `scripts/6_train_lerobot.py` completed the editable installs for `lerobot` and `lerobot_policy_hlrp` and handed off to `lerobot-train`.
