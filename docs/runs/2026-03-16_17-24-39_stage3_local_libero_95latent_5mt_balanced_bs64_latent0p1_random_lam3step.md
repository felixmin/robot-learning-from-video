# Run Note: 2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step

## Meta

- Date: 2026-03-16
- Status: completed
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

- Launched locally at `2026-03-16 17:31:28 CET` according to `unified.log` and completed at `2026-03-17 13:19:32 CET` (`~19h48m` wall time).
- Run directory resolved as `/mnt/data/workspace/runs_root/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step`; checkpoints were saved at `020000`, `040000`, `060000`, `080000`, and `100000`, with `last -> 100000`.
- Optimization stayed stable through 100k steps: the final W&B summary reports `train/loss=0.001305`, `action_loss=0.001282`, `latent_loss=0.000236`, `grad_norm=0.0292`, and `lr=2.5e-05`.
- Final train-side summary at `100k`: `samples=6.4M`, `episodes=39621.89`, `epochs=23.40`, `action_supervised_fraction=0.4375`, `latent_supervised_fraction=0.890625`.
- In-training `libero_10` evaluation stayed weak and non-monotonic across saved checkpoints: `20k 1%`, `40k 2%`, `60k 6%`, `80k 2%`, `100k 2%` success, with final `avg_sum_reward=0.02`.
- The main ablation result is negative: compared with the otherwise similar balanced run `2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`, this random-Stage1 variant reached a similarly low training loss but far worse control quality (`2%` vs `20%` in-training `libero_10` success at `100k`). This comparison is inferred from the earlier documented run note.
- No crash markers appeared in the local logs; the failure mode here is poor policy quality rather than optimizer instability or launch/runtime errors.

## Rollout Evaluation

### 2026-03-17 local rollout results

- Rollout run: `2026-03-17_13-48-10_stage3_rollout_local_from_2026-03-16_17-24-39_stage3_local_random_lam3step_bs1`
- Checkpoint: `lerobot/checkpoints/100000/pretrained_model` (step 100k)
- Launched locally at `2026-03-17 13:42:57 CET` and completed at `2026-03-17 14:49:03 CET` according to `unified.log` (`~66m` wall time).
- Overall rollout result: `27.25%` success over `400` episodes (`40` tasks x `10` episodes), with `avg_sum_reward=0.2725` and `avg_max_reward=0.2725`.
- Group-level success split: `libero_spatial 43%`, `libero_object 25%`, `libero_goal 36%`, `libero_10 5%`.
- Task outcomes were uneven: `17` tasks had zero successes, `22` were partial-success tasks, and only `1` achieved `10/10` success.
- `libero_spatial` was the strongest group, while `libero_10` remained extremely weak despite the respectable spatial/goal recovery.
- Compared with the earlier balanced run `2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1`, this random-Stage1 variant underperformed clearly on full rollouts (`27.25%` overall here vs `35.8%` there), with the largest drop on `libero_10` (`5%` here vs `15%` there). This comparison is inferred from the earlier documented run note.
- Outputs include `eval_info.json`, `400` rollout videos under `videos/`, and the launcher log in this rollout run directory.

```bash
conda run -n lerobot python scripts/7_rollout_lerobot.py \
  experiment=stage3_rollout_local \
  experiment.name=2026-03-17_13-48-10_stage3_rollout_local_from_2026-03-16_17-24-39_stage3_local_random_lam3step_bs1 \
  logging.runs_dir=/mnt/data/workspace/runs_root/runs/2026-03-17_13-48-10_stage3_rollout_local_from_2026-03-16_17-24-39_stage3_local_random_lam3step_bs1 \
  lerobot_eval.policy_path=/mnt/data/workspace/runs_root/runs/2026-03-16_17-24-39_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1_random_lam3step/lerobot/checkpoints/100000/pretrained_model \
  lerobot_eval.eval_batch_size=1 \
  'lerobot_eval.extra_args=["--env.max_parallel_tasks=1"]'
```

| Scope | Success % | Successes | Episodes | Avg Sum Reward | Avg Max Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 27.25 | 109 | 400 | 0.2725 | 0.2725 |
| libero_spatial | 43.0 | 43 | 100 | 0.43 | 0.43 |
| libero_object | 25.0 | 25 | 100 | 0.25 | 0.25 |
| libero_goal | 36.0 | 36 | 100 | 0.36 | 0.36 |
| libero_10 | 5.0 | 5 | 100 | 0.05 | 0.05 |

| Task Outcome Bucket | Count |
| --- | ---: |
| 0/10 success tasks | 17 |
| Partial-success tasks | 22 |
| 10/10 success tasks | 1 |
