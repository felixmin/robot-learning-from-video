# LAQ Artifact v1 Plan (Design Only)

This document defines the LAQ artifact contract to implement later.
No LAQ artifact code is implemented in this change.

## Goals

1. Export a strict, versioned LAQ artifact for Stage-2 and Stage-3 consumers.
2. Store complete LAQ weights (encoder + VQ + decoders/teachers) in artifact payload.
3. Allow load-time pruning or partial materialization (`encoder_vq_only`) without changing stored artifact.
4. Enforce compatibility checks before loading to avoid silent drift.
5. Keep the interface extensible for future multi-camera LAQ models.

## Scope

In scope (future implementation):
1. `packages/laq/artifact.py` with save/load utilities and manifest dataclass.
2. Stage-1 export path in `scripts/2_train_laq.py`.
3. Stage-2/Stage-3 loader paths updated to consume artifact instead of raw Lightning checkpoints.
4. Strict tests for schema/version/load compatibility.

Out of scope in this document:
1. Any checkpoint migration tooling.
2. Backward compatibility support for old artifact/checkpoint formats.

## Artifact Structure

Single `.pt` payload:
1. `schema_version`: fixed string, e.g. `laq.v1`.
2. `manifest`: JSON-serializable metadata.
3. `model_state_dict`: full LAQ model state dict (complete weights).
4. `extra`: optional structured block for future extensions (kept empty in v1).

## Manifest Fields (v1)

Core identity:
1. `schema_version`
2. `source_run_dir`
3. `source_global_step`
4. `created_at_utc`

Model contract:
1. `image_size` (H, W)
2. `channels`
3. `patch_size`
4. `codebook_size`
5. `code_seq_len`
6. `quant_dim`
7. `encoder_arch` keys required to instantiate exact LAQ model

Flow/decoder contract:
1. `flow_enabled`
2. `flow_model`
3. `decoder_flags` (`dino`, `pixel`, `aux`, `flow`)

Future multi-camera extension hooks:
1. `camera_contract` object
2. `num_expected_views`
3. `camera_fusion_mode`

The v1 implementation can set a single-view camera contract while preserving these fields.

## Loading Rules

Strict only:
1. Reject unknown schema versions.
2. Reject any shape-critical mismatch between manifest and requested runtime config.
3. Reject missing/unexpected keys on strict load.

Runtime materialization modes (future):
1. `full`: load full model (for decoding/analysis/future training resumes).
2. `encoder_vq_only`: instantiate full model, strict-load full weights, then prune unneeded modules in memory.

This keeps artifacts complete while allowing lightweight runtime usage.

## Integration Plan

1. Add `LAQArtifactManifest` + `save_laq_artifact` + `load_laq_artifact`.
2. Stage-1:
   - Export artifact at checkpoint/save milestones.
   - Keep run metadata and source step in manifest.
3. Stage-2:
   - Replace direct `load_from_checkpoint` dependency with artifact loader.
   - Build provider from loaded LAQ model.
4. Stage-3:
   - Online LAQ target generation reads same artifact.
   - Enforce latent-dim and image-size compatibility at policy init.

## Test Plan

1. Roundtrip save/load preserves manifest and strict-loadable parameters.
2. Mismatch tests:
   - `image_size`, `code_seq_len`, `codebook_size`, `quant_dim`, schema mismatch.
3. Runtime mode test:
   - load `full` and `encoder_vq_only` from same artifact; verify code outputs match.
4. Integration tests:
   - Stage-2 and Stage-3 smoke load from artifact path.

## Migration Strategy

No fallback in runtime code:
1. New runs use artifact path only.
2. Historical `.ckpt` support, if needed, is handled by one-off offline conversion scripts.
