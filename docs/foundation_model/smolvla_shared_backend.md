# SmolVLA Shared Backend (Stage 2 + Stage 3)

## Overview

This backend reimplements the SmolVLA-style flow setup with a shared trunk and split heads:

- Shared trunk: SmolVLM image+language encoder (`packages/foundation/backends/smolvla_shared/model.py`)
- Latent head: flow-matching on flattened LAQ vectors `[B, S*D]`
- Real-action head: chunked flow-matching on `[B, T, A]` (Stage 3)

Quick visual references:
- `docs/foundation_model/smolvla_shared_v2_visual_guide.md`
- `docs/foundation_model/smolvla_transform_parity.md`

Backend code is split by responsibility:

- `packages/foundation/backends/smolvla_shared/config.py`
- `packages/foundation/backends/smolvla_shared/preprocess.py`
- `packages/foundation/backends/smolvla_shared/flow.py`
- `packages/foundation/backends/smolvla_shared/losses.py`
- `packages/foundation/backends/smolvla_shared/model.py`
- `packages/foundation/backends/smolvla_shared_backend.py`

## Stage 2 Training

Use:

- `config/model/foundation_smol_flow_shared.yaml`
- `config/experiment/vla_smol_flow_shared.yaml`

Stage 2 mode is `model.training_mode=latent_flow`.
Only latent-flow loss is optimized in this stage.

Stage 2 exports full action-shape metadata (`action_dim`, `action_chunk_size`) and `normalization_stats` in artifact v2 for strict Stage 3 compatibility.

Example:

```bash
python scripts/submit_job.py \
  experiment=vla_smol_flow_shared \
  cluster=lrz_x100 \
  model.laq.checkpoint=/dss/.../laq.ckpt \
  experiment.name=vla_smol_flow_shared_smoke
```

## Stage 3 LeRobot Training

New policy type:

- `hlrp_smolvla_shared`

Config:

- `config/experiment/lerobot_hlrp_smolvla_shared_smoke.yaml`

Optional Stage-2 checkpoint handoff, depending on policy init mode:

- `lerobot.init_mode=artifact`: requires `lerobot.stage2_artifact=/path/to/smolvla_shared_stage2_artifact.pt`
- `lerobot.init_mode=scratch`: requires `lerobot.stage2_artifact=null`

### Stage 2 -> Stage 3 Artifact Contract

- Schema version: `smolvla_shared.v2`
- Producer: `scripts/4_train_foundation.py` when `model.backend=smolvla_shared`
- Default output path: `<run_dir>/artifacts/smolvla_shared_stage2_artifact.pt`
- Payload:
  - `manifest`: model/flow/transform metadata (`model_name`, `torch_dtype`, `image_size`, `action_dim`, `action_chunk_size`, tokenizer/prompt settings, camera keys, flow params, source metadata)
  - `core_state_dict`: Stage-2 shared core weights (private cache keys dropped)
- Consumer: `HLRPSmolVLASharedPolicy` in Stage 3
  - Loads `policy.stage2_artifact` only
  - Invalid/mismatched artifact schema fails fast
  - Manifest/config mismatches fail fast
  - Strict state dict load (`strict=True`) with no optional-missing fallback

Example:

```bash
python scripts/submit_job.py \
  experiment=lerobot_hlrp_smolvla_shared_smoke \
  cluster=lrz_x100 \
  lerobot.stage2_artifact=/dss/.../artifacts/smolvla_shared_stage2_artifact.pt \
  experiment.name=lerobot_hlrp_smolvla_shared_smoke
```

## Stage 3 Rollout

Rollout/eval entrypoint:

- `scripts/7_rollout_lerobot.py`

Experiment config:

- `config/experiment/lerobot_hlrp_smolvla_shared_rollout.yaml`

Required field:

- `lerobot_eval.policy_path` (path to `pretrained_model` dir)

Example:

```bash
python scripts/submit_job.py \
  experiment=lerobot_hlrp_smolvla_shared_rollout \
  cluster=lrz_x100 \
  lerobot_eval.policy_path=/dss/.../checkpoints/000050/pretrained_model \
  experiment.name=lerobot_hlrp_smolvla_shared_rollout
```
