# Stage 2 -> Stage 3 Interface Map (SmolVLA Shared v2)

## Summary

Stage 3 supports two strict initialization modes:

- `init_mode=artifact`: load Stage-2 artifact (strict contract checks).
- `init_mode=scratch`: do not load Stage-2 artifact.

- Stage 2 (`scripts/4_train_foundation.py`) writes:
  - `<run_dir>/artifacts/smolvla_shared_stage2_artifact.pt`
- Stage 3 (`scripts/6_train_lerobot.py`) forwards:
  - `--policy.init_mode=artifact|scratch`
  - `--policy.stage2_artifact=...` (artifact mode only)
- LeRobot policy (`HLRPSmolVLASharedPolicy`) loads artifact with strict checks:
  - architecture + transform config compatibility checks
  - strict state dict loading (`strict=True`, no optional missing-head fallback)
- Stage 2/3 adapters use canonical v2 batch fields only (`image_streams`, `image_padding_masks`, `task_text`, `state`, `action_is_pad`).
- Stage 3 requires camera validity masks via `<camera_key>_is_pad`.
- Stage 3 accepts action pad mask key `action_is_pad` and alias `actions_id_pad`; if both are present they must be identical.
- Action tensors are strict: `[B,T,A]` with `T == chunk_size`, except `[B,A]` only when `chunk_size == 1`.

## Artifact Contract

Schema version:
- `smolvla_shared.v2`

Payload keys:
- `schema_version`
- `manifest`
- `core_state_dict`

`manifest` fields:
- `schema_version`
- `model_name`
- `torch_dtype`
- `image_size`
- `action_dim`
- `action_chunk_size`
- `code_seq_len`
- `latent_vector_dim`
- `tokenizer_max_length`
- `pad_language_to`
- `system_prompt`
- `max_state_dim`
- `camera_keys`
- `flow_hidden_dim`
- `flow_steps`
- `min_period`
- `max_period`
- `time_beta_alpha`
- `time_beta_beta`
- `source_backend`
- `source_training_mode`
- `source_run_dir`
- `source_global_step`
- `normalization_stats`

## Failure Behavior

Fail-fast behavior:
- Missing artifact path -> `FileNotFoundError`
- Bad schema/version -> `ValueError`
- Malformed payload/state dict -> `TypeError`/`KeyError`
- Stage-3 config/manifest mismatch -> `ValueError`
- Any strict load key mismatch -> `RuntimeError`

## Minimal Flow

1. `init_mode=artifact`:
   run Stage 2 with `model.backend=smolvla_shared`, `model.flow.action_dim`, and `model.flow.action_chunk_size`.
2. `init_mode=artifact`:
   use `lerobot.stage2_artifact=<artifact_path>` and run Stage 3.
3. `init_mode=scratch`:
   set `lerobot.stage2_artifact=null` and run Stage 3.

## TODO

- Stage-2 `ACTIONS`/`MULTITASK` currently requires `batch["action_is_pad"]` in the OXE batch path (`packages/foundation/vla_backend_module.py`).
  Add `action_is_pad` emission in the Stage-2 OXE data path/collate before enabling these modes broadly.
