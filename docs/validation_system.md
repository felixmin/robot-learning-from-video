# Validation System Architecture

## Overview

LAQ training uses a bucket-aware validation callback with strategy plugins:

- Batches from the normal validation dataloader are processed once.
- Samples are routed into a global cache and optional per-bucket caches.
- Strategies run at validation end, each on either global cache or selected buckets.

This is implemented with Lightning callbacks and strategy classes, not with a separate validation trainer.

## Core Components

### Validation callback

`packages/laq/callbacks.py` defines `ValidationStrategyCallback`.

Responsibilities:
- clear caches at validation start
- extract per-sample metadata from batch
- optionally compute code indices if any active strategy needs them
- route each sample to global cache and all matching bucket caches
- run strategies at epoch end with bucket-aware metric suffixes

### Cache + strategy base

`packages/laq/validation/core.py` provides:
- `ValidationCache`
- `BucketConfig`
- `ValidationStrategy`

Important behavior:
- frame storage is bounded (`max_samples`)
- code history (`all_codes`) is unbounded/lightweight for histogram-style strategies
- strategies declare frequency (`every_n_validations`), minimum data, metadata requirements, and optional `buckets`

### Strategy factory

`packages/laq/validation/factory.py` maps config entries to strategy classes.

Current registry keys:
- `basic` / `basic_visualization`
- `latent_transfer`
- `codebook_histogram`
- `sequence_histogram`
- `all_sequences_histogram`
- `codebook_embedding`
- `sequence_examples`
- `action_token_scatter`
- `action_sequence_scatter`
- `top_sequences_scatter`
- `state_sequence_scatter`
- `flow_visualization`

## Bucket Routing Model

Buckets are configured under `training.validation.buckets`.

Example:

```yaml
training:
  validation:
    buckets:
      language_table:
        filters: {dataset_name: "language_table"}
        max_samples: 256
      bridge:
        filters: {dataset_name: "bridge"}
        max_samples: 256
```

Routing model:
- each sample always goes to global cache
- sample also goes to each bucket whose filters match metadata
- strategies with `buckets: []` (or omitted) run on global cache
- strategies with `buckets: ["..."]` run on those bucket caches only

## Validation Lifecycle

1. Validation starts: callback clears caches.
2. Each batch:
- extract metadata
- optionally compute codes
- route to caches
3. Validation end:
- increment strategy counters
- set fixed sample subsets once
- run strategies whose cadence is due and whose data requirements pass `can_run`

## Current Notes and Constraints

- Bucket-specific dataloaders were removed from the DataModule. Routing is callback-based.
  - `train_bucket_dataloader` and `val_bucket_dataloader` intentionally raise `NotImplementedError` in `packages/common/data.py`.
- Split mode `fixed_count` targets frame-pair counts per dataset (not raw scene counts).
- `LatentTransferStrategy` uses `pl_module.encode_latents(...)` and `decode_with_latents(...)`.

## Configuration Surface

Primary training validation config lives in:
- `config/training/laq_optimizer.yaml` under `training.validation`

Main knobs:
- `check_interval`
- `limit_batches`
- `num_fixed_samples`
- `num_random_samples`
- `max_cached_samples`
- `buckets`
- `strategies.*` (enabled, cadence, strategy-specific params)

## Data Split Interaction

Validation strategy caching runs on whatever val split the DataModule produced.
Current split behavior for LeRobot-v3-backed Stage 1/2 is implemented in:
- `packages/common/lerobot_v3_data.py`

## Implementation Files

- `packages/laq/callbacks.py`
- `packages/laq/validation/core.py`
- `packages/laq/validation/factory.py`
- `packages/laq/validation/visualization.py`
- `packages/laq/validation/analysis.py`
- `packages/laq/validation/scatter.py`
- `packages/laq/validation/flow.py`
- `config/training/laq_optimizer.yaml`

Training-time validation is intentionally lightweight and periodic during training.
