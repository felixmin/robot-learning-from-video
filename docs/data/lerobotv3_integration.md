# LeRobot v3 Integration Findings for HLRP Stage 2

Last updated: 2026-03-02

## Scope

This note captures the current feasibility analysis for adding LeRobot v3 datasets into HLRP Stage 2 training, with a focus on mixing them with the existing OXE local indexed pipeline and keeping a clean migration path toward a future LeRobot-only data stack.

The main concrete dataset inspected during this analysis was the local LIBERO snapshot:

- `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e`

## Executive summary

1. Adding LeRobot v3 data to Stage 2 is feasible.
2. The current Stage 2 stack does not support it yet because it is hard-wired to the OXE local indexed backend.
3. The clean design is:
   - keep one shared Stage 2 batch contract
   - implement a standalone `lerobot_v3` loader that emits that contract
   - add a thin `mixed_sources` wrapper that mixes `oxe_local_indexed` and `lerobot_v3`
4. If done this way, OXE can be removed later with minimal churn by switching config from `mixed_sources` to `lerobot_v3`.
5. The main code-level compatibility gap is chunked action support in parts of Stage 2 that still assume actions are `[B, A]` instead of `[B, T, A]`.

## What the current Stage 2 stack expects

Stage 2 currently only accepts `data.backend='oxe_local_indexed'`:

- [`packages/common/data_factory.py`](../../packages/common/data_factory.py)

The current OXE local indexed implementation builds batches around:

- `frames`
- `language`
- `initial_state`
- `action`
- optional `action_is_pad`
- metadata such as `dataset_name`, `episode_id`, `frame_idx`

Relevant implementation points:

- [`packages/common/data.py`](../../packages/common/data.py)
- [`packages/common/adapters/openx_local_indexed_full.py`](../../packages/common/adapters/openx_local_indexed_full.py)
- [`packages/foundation/vla_backend_module.py`](../../packages/foundation/vla_backend_module.py)
- [`packages/foundation/online_laq.py`](../../packages/foundation/online_laq.py)

Current OXE indexed samples are built from local tar shards with pickled episodes, not parquet:

- [`packages/common/adapters/openx_local_indexed_full.py`](../../packages/common/adapters/openx_local_indexed_full.py)

This means LeRobot v3 cannot be dropped into the current backend without adding a new loader path.

## What LeRobot v3 looks like

The installed LeRobot package in this environment is:

- `lerobot==0.4.2`

The main dataset classes are:

- `lerobot/datasets/lerobot_dataset.py`

Important classes:

- `LeRobotDataset`
- `LeRobotDatasetMetadata`
- `MultiLeRobotDataset`

### Dataset layout

The inspected LIBERO snapshot is a valid LeRobot v3 dataset:

- `meta/info.json`
- `meta/stats.json`
- `meta/tasks.parquet`
- `meta/episodes/...`
- `data/chunk-000/file-*.parquet`

Confirmed metadata from LIBERO:

- `codebase_version: v3.0`
- `robot_type: panda`
- `fps: 10.0`
- `total_episodes: 1693`
- `total_frames: 273465`
- `total_tasks: 40`

### Feature schema in the inspected LIBERO snapshot

Confirmed features:

- `observation.images.image`
- `observation.images.image2`
- `observation.state` with shape `[8]`
- `action` with shape `[7]`
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`

In this particular LIBERO snapshot, images are stored inline in parquet as image features and decode locally without requiring mp4 video files.

### What `LeRobotDataset` returns

Using `LeRobotDataset(..., root=<snapshot_dir>)` against the local LIBERO snapshot works offline and returns items containing:

- `observation.images.image`
- `observation.images.image2`
- `observation.state`
- `action`
- `task`
- `task_index`
- `episode_index`
- `frame_index`
- `timestamp`

Observed types for one item:

- `observation.images.image`: `torch.Tensor` with shape `(3, 256, 256)`
- `observation.images.image2`: `torch.Tensor` with shape `(3, 256, 256)`
- `observation.state`: `torch.Tensor` with shape `(8,)`
- `action`: `torch.Tensor` with shape `(7,)`
- `task`: `str`

Important operational detail:

- To use a pre-downloaded local snapshot cleanly, `root` must point directly at the snapshot directory, not at a higher-level Hugging Face cache root.

## Why LeRobot v3 is a good fit for Stage 2

LeRobot already supports temporal querying with padding masks through `delta_timestamps`.

This is the key capability needed for Stage 2 because we need:

1. frame pairs for LAQ supervision
2. optional action chunks for action-flow training
3. valid/padded mask information near episode boundaries

Confirmed behavior from local testing with LIBERO:

- requesting `observation.images.image` at `[0.0, 0.5]` returns a tensor of shape `(2, 3, 256, 256)`
- requesting `action` over a future chunk returns shape `(T, 7)`
- LeRobot also returns `action_is_pad` with shape `(T,)`
- near the end of an episode, `action_is_pad` becomes partially true as expected

This is exactly the behavior needed for chunked action supervision.

## Recommended architecture

The clean architecture is:

1. format-specific loaders
2. one shared Stage 2 batch contract
3. an optional mixer wrapper

### Shared Stage 2 batch contract

Both OXE and LeRobot loaders should emit the same normalized sample/batch schema:

- `frames`
- `language`
- `initial_state`
- `action`
- `action_is_pad`
- `dataset_name`
- `episode_id`
- `frame_idx`

Suggested conventions:

- `frames`: `[3, 2, H, W]` or batch-stacked equivalent expected by current Stage 2 code
- `language`: task/instruction string
- `initial_state`: `[S]`
- `action`: `[A]` for non-chunked modes or `[T, A]` for chunked action modes
- `action_is_pad`: `[T]` bool when chunked actions are present

All data-format-specific logic should stay inside the source loader, not inside the training module and not inside the mixer.

### Loader split

Recommended `data.backend` split:

1. `oxe_local_indexed`
   - existing backend
2. `lerobot_v3`
   - new standalone backend
   - fully usable on its own
3. `mixed_sources`
   - thin wrapper that mixes `oxe_local_indexed` and `lerobot_v3`

This keeps the migration path clean:

- now: `mixed_sources`
- later: `lerobot_v3`
- eventual cleanup: delete OXE backend and mixer if desired

## How the LeRobot loader should work

### Input configuration

The LeRobot loader should take per-dataset entries such as:

- `repo_id`
- `root`
- `train_split`
- `val_split`
- `weight`
- `pair_offset_steps`
- `action_chunk_size`
- camera key selection

### Temporal querying

For a Stage 2 dataset entry, build `delta_timestamps` from dataset FPS.

Example for `fps=10`, `pair_offset_steps=5`, `action_chunk_size=50`:

- image deltas: `[0.0, 0.5]`
- state deltas: `[0.0]`
- action deltas: `[0.0, 0.1, ..., 4.9]`

Then map the queried item to the shared Stage 2 schema:

- `task -> language`
- `observation.state -> initial_state`
- `observation.images.image -> first/last frame pair`
- `action -> future action chunk`
- `action_is_pad -> chunk validity mask`

### Camera handling

For LIBERO specifically:

- image keys: `observation.images.image`, `observation.images.image2`
- video keys: none

The current Stage 2 code path still collapses input into a single image stream in some places, so the first LeRobot implementation should pick a single camera consistently unless multi-camera support is intentionally expanded at the same time.

## Mixing behavior

Two mixing strategies are reasonable.

### 1. Sample-level mixing

Behavior:

- the mixer chooses a source for each sample draw
- one batch may contain a mix of OXE and LeRobot samples

Pros:

- closest to the current OXE sampling style
- easy to get started

Cons:

- noisier batch composition
- harder to debug dataset-specific behavior

### 2. Batch-level mixing

Behavior:

- the mixer chooses one source for a whole batch
- each batch is pure OXE or pure LeRobot

Pros:

- easier debugging
- cleaner per-dataset metrics
- easier to reason about temporary schema differences

Cons:

- slightly more sampling code

Recommended bring-up order:

1. implement batch-level pure-batch mixing first
2. add sample-level mixing later only if needed

## Main compatibility gap to fix

The biggest Stage 2 compatibility issue is action tensor rank.

Today, some Stage 2 helpers still assume actions are rank 2:

- [`packages/foundation/online_laq.py`](../../packages/foundation/online_laq.py)

`extract_oxe_actions` currently expects `[B, A]`.

That is fine for current OXE latent-only style supervision, but LeRobot chunked actions naturally produce `[B, T, A]`.

The backend core already supports chunked actions:

- [`packages/foundation/backends/smolvla_shared/model.py`](../../packages/foundation/backends/smolvla_shared/model.py)
- [`packages/foundation/backends/smolvla_shared/input_transform.py`](../../packages/foundation/backends/smolvla_shared/input_transform.py)

So the required change is not a model redesign. It is mainly an input-path normalization change so Stage 2 consistently accepts both:

- `[B, A]`
- `[B, T, A]`

with correct `action_is_pad`.

## Should we parse raw LeRobot parquet ourselves?

Possible, but not recommended as the first path.

### Use `LeRobotDataset`

Pros:

- already handles v3 metadata
- already handles image decoding
- already handles temporal querying
- already provides `*_is_pad`
- already supports standalone use later

Cons:

- needs adapter code to map into HLRP Stage 2 schema

### Parse raw parquet directly

Pros:

- full control
- potentially tighter optimization later

Cons:

- reimplements LeRobot behavior already provided by the library
- more maintenance
- more room for subtle bugs around episode boundaries and padding

Recommendation:

- first implementation should use `LeRobotDataset`
- only consider raw parquet parsing later if profiling shows a real bottleneck

## Migration path toward LeRobot-only Stage 2

The transition plan should be:

1. add a real `lerobot_v3` Stage 2 loader
2. add a thin `mixed_sources` wrapper
3. keep training code consuming only the shared Stage 2 batch contract
4. later switch configs from `mixed_sources` to `lerobot_v3`
5. delete OXE loader after validation if desired

If designed this way, removing OXE later should be mostly a config and cleanup task, not a training-stack rewrite.

## Feasibility verdict

### Mixing OXE and LeRobot now

Feasible:

- yes

Expected difficulty:

- medium

Main work items:

1. new LeRobot datamodule/adapter
2. source mixer wrapper
3. action rank normalization cleanup
4. config plumbing and tests

### Going LeRobot-only later

Feasible:

- yes

This is significantly easier if the mixed implementation is built around:

- standalone `lerobot_v3`
- standalone `oxe_local_indexed`
- dumb `mixed_sources` wrapper

instead of baking LeRobot logic into the current OXE loader.

## Concrete recommendation

Implement the following, in this order:

1. `data.backend=lerobot_v3`
2. a LeRobot Stage 2 loader that emits the shared Stage 2 schema
3. `data.backend=mixed_sources`
4. pure-batch alternating source mixing first
5. optional sample-level mixing later
6. shared action-input cleanup so `[B, T, A]` is supported end to end

Do not:

1. jam LeRobot parsing into the OXE tar loader
2. put format-specific conditionals into the Stage 2 Lightning module
3. make the mixer responsible for schema conversion

That would make later OXE removal much more expensive.
