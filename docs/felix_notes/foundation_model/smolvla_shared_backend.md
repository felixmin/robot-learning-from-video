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

- `config/model/stage2_smol_flow_shared.yaml`
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

Current experiment configs:

- `config/experiment/stage3_hlrp_libero_action_scratch.yaml`
- `config/experiment/stage3_hlrp_libero_multitask_scratch.yaml`
- `config/experiment/stage3_hlrp_libero_multitask_scratch_cluster.yaml`
- `config/experiment/stage3_hlrp_libero_multitask_scratch_local.yaml`

Optional Stage-2 checkpoint handoff, depending on policy init mode:

- `lerobot.policy.init_mode=artifact`: requires `lerobot.policy.stage2_artifact=/path/to/smolvla_shared_stage2_artifact.pt`
- `lerobot.policy.init_mode=scratch`: requires `lerobot.policy.stage2_artifact=null`

### Stage 2 -> Stage 3 Artifact Contract

- Schema version: `smolvla_shared.v2`
- Producer: `scripts/4_train_stage2_policy.py` when `model.backend=smolvla_shared`
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
  experiment=stage3_hlrp_libero_action_scratch \
  cluster=lrz_x100 \
  lerobot.policy.init_mode=artifact \
  lerobot.policy.stage2_artifact=/dss/.../artifacts/smolvla_shared_stage2_artifact.pt \
  lerobot.steps=50 \
  lerobot.batch_size=2 \
  lerobot.eval.freq=10 \
  lerobot.log_freq=10 \
  lerobot.save_freq=1000 \
  experiment.name=lerobot_hlrp_smolvla_shared_smoke
```

## Stage 3 Rollout

Rollout/eval entrypoint:

- `scripts/7_rollout_lerobot.py`

Experiment config:

- `config/experiment/stage3_rollout_local.yaml`
- `config/experiment/stage3_rollout_cluster.yaml`

Required field:

- `lerobot_eval.policy_path` (path to the exported `pretrained_model` dir inside a saved checkpoint)

Examples of valid policy paths:

- `/dss/.../checkpoints/000050/pretrained_model`
- `/mnt/data/.../lerobot/checkpoints/last/pretrained_model`

Example:

```bash
conda run -n lerobot python scripts/7_rollout_lerobot.py \
  experiment=stage3_rollout_local \
  lerobot_eval.policy_path=/mnt/data/.../lerobot/checkpoints/last/pretrained_model \
  experiment.name=stage3_rollout_local_example
```

Cluster example:

```bash
python scripts/submit_job.py \
  experiment=stage3_rollout_cluster \
  cluster=lrz_x100 \
  lerobot_eval.policy_path=/dss/.../checkpoints/000050/pretrained_model \
  experiment.name=stage3_rollout_cluster_example
```

## Known Open Item

- Stage-2 `ACTIONS`/`MULTITASK` on the OXE path still depends on `action_is_pad` being emitted by the Stage-2 data adapter/collate. Until that is wired broadly, the shared Stage-2 backend is fully ready for `latent_flow`, while action-supervised Stage-2 usage should be treated as opt-in and data-path-dependent.

## Implementation Note

- The original redesign notes mentioned dedicated Stage-3 processor classes for newline/tokenization parity. The current implementation centralizes that logic in `packages/foundation/backends/smolvla_shared/input_transform.py` instead. Treat that module as the source of truth for language/image/state/action transform semantics.
