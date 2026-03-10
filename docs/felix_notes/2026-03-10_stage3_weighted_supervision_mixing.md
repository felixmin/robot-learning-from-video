# Stage 3 Weighted Supervision Mixing

Implemented the Stage 3 supervision-mix path from [docs/plan/2026-03-10_16-13-40_stage3_weighted_supervision_mixing.md](/mnt/data/workspace/code/high-level-robot-planner/docs/plan/2026-03-10_16-13-40_stage3_weighted_supervision_mixing.md).

## What Changed

- Stage 3 supervision ownership moved from policy-side ratio masks to dataset-provided booleans.
- Vendored `lerobot/` now supports `dataset.mix_path` with logical sources, source weights, episode subsets, and per-source supervision modes.
- Stage 3 dataset presets now point to committed mix YAMLs under `config/stage3_dataset_mix/`.
- Fallback normalization-stat repair is mix-aware and rebuilds from the mix definition rather than `dataset.repo_id` alone.

## Migration

Removed Stage 3 policy config knobs:

- `lerobot.policy.action_subset_ratio`
- `lerobot.policy.action_subset_key`
- `lerobot.policy.latent_scope`

Use dataset-side mix presets instead:

- [config/stage3_dataset/libero.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/stage3_dataset/libero.yaml)
- [config/stage3_dataset/libero_5pct.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/stage3_dataset/libero_5pct.yaml)
- [config/stage3_dataset/libero_5pct_latent_rest_natural.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/stage3_dataset/libero_5pct_latent_rest_natural.yaml)
- [config/stage3_dataset/libero_5pct_latent_rest_balanced.yaml](/mnt/data/workspace/code/high-level-robot-planner/config/stage3_dataset/libero_5pct_latent_rest_balanced.yaml)

## Validation

Focused local validation passed for:

- Stage 3 launcher/config/policy/checkpoint tests in repo `tests/`
- vendored `lerobot` mixed-dataset and sampler tests

Remaining rollout risk: distributed sampler behavior still needs a real multi-process smoke before cluster use.
