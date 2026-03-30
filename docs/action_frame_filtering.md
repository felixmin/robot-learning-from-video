# Action-Frame Filtering (Low Motion / Low Action)

This repo supports configurable per-anchor filtering in the shared LeRobot-v3 pipeline (`packages/common`).

## Where it runs

- Source compile stage in `LeRobotSingleSource.compile`.
- Filtered anchors are materialized once, then reused by samplers in `packages/common/lerobot_v3_sampler.py`.

## Config keys

Configure under `data.filtering` (defaults in `config/data/_lerobot_v3_mix_base.yaml`):

- `enabled`: global on/off.
- `mode`: `none | motion | action | both`.
- `apply_at_sampling`: if true, sampler uses filtered anchors.
- `trim_episode_ends`: apply motion-based endpoint trimming.

Motion section (`data.filtering.motion`):

- `enabled`, `method` (`frame_diff | sparse_flow | two_stage`)
- `frame_gap` (`null` => inferred from requested image deltas)
- `decode_backend` (`null | pyav | torchcodec`; `null` uses dataset backend)
- `decode_chunk_size` (number of anchors decoded together per episode chunk)
- `prefetch_next_episode` (warm up decode for next episode while current episode is scored)
- `prefetch_chunk_size` (warmup anchor count for next-episode prefetch)
- `device` (`cpu | cuda` for motion scoring tensor ops)
- `aggregate_all_cameras` (compute motion on all available camera keys for this source)
- `aggregate_reduce` (`mean | max`) to build one aggregate motion score from per-camera scores
- `resize_short_side`, `blur_kernel`, `diff_pixel_threshold`
- `smoothing_window`, `consecutive_active_k`
- `low_threshold`, `high_threshold`, `use_hysteresis`

Action section (`data.filtering.action`):

- `enabled`, `threshold`, `exclude_dims`, `delta_dims`
- `chunk_size`, `chunk_reduce` (`max | mean`)
- `min_nonzero_ratio`
- Time spacing: action score uses consecutive dataset steps (`anchor+0, +1, ...`), so spacing is 1 dataset step between points.
- Practical timeframe: approximately `chunk_size` video frames and `chunk_size / fps` seconds (for example at 15 FPS, `chunk_size=3` spans about 3 frames and 0.20 s).

Semantics:
- `exclude_dims` removes those action vector dimensions before L2 norm scoring.
- Example: with `exclude_dims=[6]`, action dim index 6 (7th component, often gripper) is ignored in action score computation.
- `delta_dims` scores selected action dimensions by temporal change (`x[t]-x[t-1]`) inside the chunk instead of absolute value.
- Example: with `delta_dims=[6]` and `chunk_size>=2`, a binary gripper channel (0/1) contributes mainly when it toggles, not when it stays open/closed.
- `chunk_size` is the number of consecutive action steps scored per anchor.
- Example: `chunk_size=3` means score is computed from actions at `anchor+0`, `anchor+1`, `anchor+2`.
- Implementation detail: action scoring is computed on contiguous indices in `_action_scores_for_episode_batched` (`packages/common/anchor_filtering.py`), not on sparse custom action deltas.
- `action.chunk_reduce` operates on the action chunk only (time/chunk dimension), not cameras.
- `chunk_reduce=max`: keep if any step in the chunk has high action norm (less aggressive drop).
- `chunk_reduce=mean`: keep only if sustained action across the chunk (more aggressive drop).
- Camera aggregation is controlled separately by `motion.aggregate_all_cameras` and `motion.aggregate_reduce`.

Cache section (`data.filtering.cache`):

- `enabled`
- `reuse_if_config_unchanged`
- `force_recompute`

## Filtering Config Locations

Action-frame filtering is configured separately for Stage 1/2 and Stage 3.

- Stage 1/2 filtering lives in data configs under `config/data/*.yaml`.
- Example: `config/data/lsy_teleop_only.yaml` -> `data.filtering`.
- Stage 3 filtering lives in stage-3 dataset configs under `config/stage3_dataset/*.yaml`.
- Example: `config/stage3_dataset/lsy_teleop_full_multitask.yaml` -> `lerobot.dataset.filtering`.
- Stage 3 source list lives in `config/stage3_dataset_mix/*.yaml` and can optionally override filtering per source via `sources[i].filtering`.
- Merge order at runtime is: Stage 3 global dataset filtering, then per-source override (if present).

Implication: Stage 3 does not automatically inherit Stage 1 filtering values unless you set matching values explicitly.

For filtering semantics and cache/debugging details, see `docs/action_frame_filtering.md`.


## Mixed datasets behavior

- Sources with actions: action score is computed and can be used in `action` / `both` mode.
- Sources without actions (`action_key: null`): action criterion is skipped (anchors are not dropped because action is missing).
- Per-source override is supported via `dataset.lerobot.sources[i].filtering`.

## Camera selection behavior

- If `motion.aggregate_all_cameras=false` (default), motion score uses the source's requested primary camera.
- If `motion.aggregate_all_cameras=true`, motion score is computed over camera keys from the source `camera_map` in dataset YAML and aggregated using `motion.aggregate_reduce`.
- If `camera_map` has no keys, it falls back to all camera keys in dataset metadata.
- This camera aggregation affects motion keep/drop and trim logic.
- Action filtering does not use camera keys.

Implication for `lsy_teleop_only`:
- Current config defines one source with multi-camera `camera_map`.
- With `motion.aggregate_all_cameras=true`, filtering aggregates over those mapped camera keys.

## Cache location and reuse

- Sidecar cache files are written to:
  - `<dataset_root>/meta/hlrp_action_frame_filter_cache/<split>_<camera_tag>_<fingerprint>.npz`
- Fingerprint includes relevant filtering config + source/delta settings + episode candidate bounds.
- Matching fingerprint => cache hit, scoring is skipped.
- Cache payload also includes `fingerprint_payload_json` and `filtering_config_json` for provenance/debugging.

### Stage 3 usage

- Stage 3 mixed-dataset sources can use the same filtering backend and cache files as Stage 1.
- Global defaults for Stage 3 live under `lerobot.dataset.filtering`.
- Per-source override is available in `config/stage3_dataset_mix/*.yaml` via `source.filtering`.
- Camera selection in Stage 3 mix sources:
  - `camera_map` or `camera_keys` can be set per source.
  - if neither is set, all dataset cameras are used.
  - `camera_map` and `camera_keys` are mutually exclusive.
- Stage 3 logs cache behavior per source:
  - cache hit: `[mixed-filter] cache_hit ...`
  - cache regeneration: `[mixed-filter] regenerated cache ...`

Notes:
- `camera_tag` is camera-specific for single-camera mode (for example `fpv`) and `allcamsN` for aggregated multi-camera mode.
- `train_*.npz` and `val_*.npz` are always separate files.
- With multi-source configs (for example one source per camera), you will have multiple files per split.

Quick way to locate cache files:

```bash
find cache/huggingface/lerobot -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
find ~/.cache -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
find / -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
```

Default local path pattern:

- `cache/huggingface/lerobot/<ORG>/<DATASET>/meta/hlrp_action_frame_filter_cache/`
- Example in this repo: `cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache/`

Why multiple files appear:

- Train/val are always separate files (`train_*.npz` and `val_*.npz`).
- Each file name includes a fingerprint hash; changed thresholds/config/candidate bounds create a new file.
- Old files are intentionally kept; reuse picks the file matching the current fingerprint.
- Multiple train files for one camera/split usually mean multiple filtering variants were run over time.

Fast decision-only reuse:

- If score-producing settings match but only decision settings changed (for example thresholds/mode/trim), filtering can reuse cached scores and only recompute keep/drop masks.
- Logs show this as `cache=score-hit` (or `Action-frame filtering cache score-hit ...`).
- Score-producing changes (for example frame-diff preprocessing, action chunk scoring params, camera/split/candidate bounds) still trigger full recompute.

## Debugging and tuning

### Why normalization is required

Filtering config is passed through `normalize_filtering_config(...)` before runtime use.
That function is a strict schema/merge step:

- merges global + source overrides,
- fills defaults for missing keys,
- and only forwards known keys into runtime filtering.

If a newly introduced key (for example `motion.decode_chunk_size`) is not added to normalization,
it is silently dropped and runtime falls back to default behavior.
That is why decode/prefetch/device keys were added there explicitly.

### Current decode + preprocessing pipeline

For each source split:

1. Build anchor candidates per episode (`valid_anchor_start/end`).
2. For each episode, process anchors in `motion.decode_chunk_size` chunks.
3. For each camera in the selected camera set:
   - resolve timestamps for `(t0, t1)` pairs,
   - decode frames via `_decode_frames_for_filter(...)`:
     - uses `torchcodec` decode path + `VideoDecoderCache` when available,
     - otherwise uses `decode_video_frames(..., backend=pyav/video_reader)`.
   - optional fallback: if `torchcodec` fails at runtime, log once and continue with `pyav`.
4. Motion preprocessing/scoring (`motion_scores_from_pairs`):
   - convert RGB to grayscale,
   - optional resize to `resize_short_side`,
   - optional blur (`blur_kernel`),
   - threshold pixel diffs (`diff_pixel_threshold`),
   - aggregate active pixel fraction.
5. Smooth motion score over time (`smoothing_window`) and apply hysteresis thresholds.
6. Compute batched action scores per episode (`action.chunk_size`, `action.chunk_reduce`).
7. Combine mode logic (`motion|action|both`) and optional endpoint trimming.
8. Materialize keep-mask + compact anchor arrays and write/read cache sidecar.

Prefetch behavior:

- if `motion.prefetch_next_episode=true`, one background thread warms decode on a small
  chunk of the next episode (`prefetch_chunk_size`) while current episode is scored.
- this is a decode warmup optimization; it does not change filtering semantics.

### Config-first recommendation

Put stable filtering values directly in your dataset config (for example `config/data/lsy_teleop_only.yaml`) and avoid long CLI override chains.

Use temporary CLI overrides only for:
- `training.max_steps` / debug runtime controls
- `data.filtering.cache.force_recompute=true` when intentionally regenerating caches

### Commands

1. Run a short job to create/update cache files:

```bash
python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.adapter.lerobot_v3.steps_per_epoch=20
```

2. Force cache regeneration after threshold changes:

```bash
python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.filtering.cache.force_recompute=true
```

3. Plot a single cache file:

```bash
python scripts/plot_action_frame_filter_cache.py <path/to/cache_file.npz> --episode-row 0
```

4. Save a plot image while tuning:

```bash
python scripts/plot_action_frame_filter_cache.py <path/to/cache_file.npz> --episode-row 0 --save-plot
```

Default output path with `--save-plot` is:
- `runs/debug/action_frame_filtering_ep<episode_row>.png`

5. Compare all cache files for one split in one figure:

```bash
python scripts/plot_action_frame_filter_cache.py \
  --cache-dir <dataset_root>/meta/hlrp_action_frame_filter_cache \
  --split train \
  --all-files \
  --fps 15 \
  --episode-row 0 \
  --save-path runs/debug/action_frame_filtering_all_train_ep0.png
```

The plot shows:
- per-camera `motion_raw` / `motion_smooth` traces (if present)
- aggregated `motion_raw(agg)` / `motion_smooth(<reduce>)`
- trim start/end markers
- motion threshold lines
- `action_score` (if present) + action threshold
- final `keep_mask` on a separate right y-axis (0/1)
- x-axis labels as `anchor_index` and `time_seconds`

If `--save-path` is provided, the plot is saved exactly there. Otherwise, use `--save-plot` to save to the default path.

### Recommended tuning order

1. Tune motion first:
- set `mode=motion`
- stabilize `low_threshold`, `high_threshold`, `aggregate_reduce`

2. Tune action second:
- switch to `mode=both`
- tune `threshold`, `chunk_size`, `chunk_reduce`, `min_nonzero_ratio`
- for this teleop dataset, `exclude_dims=[6]` is typically useful because gripper dominates norm scale
- if a channel should matter only on transitions (for example binary open/close), use `action.delta_dims=[<dim>]` with `chunk_size>=2`

3. Lock deployment values in dataset config:
- keep `force_recompute=false`
- run a short confirmation job and validate cache hit/miss behavior in logs

## Deployment checklist

- Filtering values live in dataset config (not ad-hoc CLI).
- `force_recompute=false` for normal runs.
- Cache files exist for both train and val splits.
- Spot-check worst episodes by plotting high-drop rows.
- If moving to another dataset config, copy filtering block and rerun one forced recompute.

## Suggested quick validation commands

```bash
conda run -n hlrp ruff check packages/common/anchor_filtering.py packages/common/action_frame_filtering.py packages/common/lerobot_v3_source.py packages/common/lerobot_v3_sampler.py scripts/plot_action_frame_filter_cache.py
conda run -n hlrp pytest -q tests/test_common.py
```

## End-to-end verification runbook

Use this exact sequence to validate functionality, cache behavior, and threshold impact.

### 1) Stage 1 teleop smoke (cache create/reuse)

```bash
conda run -n hlrp python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.adapter.lerobot_v3.steps_per_epoch=20
```

Expected:
- first run: cache miss and file generation
- second identical run: cache hit (no recompute)

### 2) Stage 1 forced cache regeneration

```bash
conda run -n hlrp python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=50 \### Filtering Config Locations

Action-frame filtering is configured separately for Stage 1/2 and Stage 3.

- Stage 1/2 filtering lives in data configs under `config/data/*.yaml`.
- Example: `config/data/lsy_teleop_only.yaml` -> `data.filtering`.
- Stage 3 filtering lives in stage-3 dataset configs under `config/stage3_dataset/*.yaml`.
- Example: `config/stage3_dataset/lsy_teleop_full_multitask.yaml` -> `lerobot.dataset.filtering`.
- Stage 3 source list lives in `config/stage3_dataset_mix/*.yaml` and can optionally override filtering per source via `sources[i].filtering`.
- Merge order at runtime is: Stage 3 global dataset filtering, then per-source override (if present).

Implication: Stage 3 does not automatically inherit Stage 1 filtering values unless you set matching values explicitly.

For filtering semantics and cache/debugging details, see `docs/action_frame_filtering.md`.

  logging.use_wandb=false \
  data.filtering.cache.force_recompute=true
```

Expected:
- cache is regenerated regardless of fingerprint match

### 3) Stage 3 teleop smoke with shared filtering backend

```bash
conda run -n lerobot python scripts/6_train_lerobot.py \
  experiment=stage3_local_teleop_stage1 \
  artifacts.lam_checkpoint_path=<path/to/stage1_checkpoint.ckpt> \
  lerobot.steps=20 \
  lerobot.batch_size=4 \
  lerobot.num_workers=2 \
  lerobot.log_freq=10 \
  lerobot.eval.freq=0 \
  logging.use_wandb=false \
  lerobot.wandb.enable=false
```

Checkpoint-free action-only smoke variant:

```bash
conda run -n lerobot python scripts/6_train_lerobot.py \
  experiment=stage3_local_teleop_stage1 \
  lerobot.policy.stage3_training_mode=action \
  lerobot.policy.lam_checkpoint_path=null \
  lerobot.policy.lam_future_frames=null \
  lerobot.steps=20 \
  lerobot.batch_size=4 \
  lerobot.num_workers=2 \
  lerobot.log_freq=10 \
  lerobot.eval.freq=0 \
  logging.use_wandb=false \
  lerobot.wandb.enable=false
```

Sample checkpoint path:
home/maxchr/repos/robot-learning-from-video/runs/2026-03-14_14-59-59_stage1_local/checkpoints/last.ckpt

Expected logs per source:
- cache hit: `[mixed-filter] cache_hit ...`
- cache regeneration: `[mixed-filter] regenerated cache ...`

### 4) Stage 3 forced regeneration override

```bash
conda run -n lerobot python scripts/6_train_lerobot.py \
  experiment=stage3_local_teleop_stage1 \
  lerobot.steps=50 \
  lerobot.eval.freq=0 \
  logging.use_wandb=false \
  lerobot.wandb.enable=false \
  lerobot.dataset.filtering.cache.force_recompute=true
```

### 5) Human dataset motion-threshold tuning

Config: `config/data/lsy_human_only.yaml`

Base run:

```bash
conda run -n hlrp python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_human_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.loader.batch_size=16 \
  data.adapter.lerobot_v3.steps_per_epoch=20
```

Threshold sweep example:

```bash
conda run -n hlrp python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_human_only \
  training.max_steps=50 \
  logging.use_wandb=false \
  data.filtering.motion.low_threshold=0.006 \
  data.filtering.motion.high_threshold=0.012 \
  data.filtering.cache.force_recompute=true
```

### 6) Visual debugging

Find cache files:

```bash
find cache/huggingface/lerobot -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
find ~/.cache -name "train_*.npz" -path "*hlrp_action_frame_filter_cache*" 2>/dev/null
```

Plot one cache file:

```bash
python scripts/plot_action_frame_filter_cache.py <path/to/cache_file.npz> --episode-row 0
```

Plot all train cache files:

```bash
python scripts/plot_action_frame_filter_cache.py \
  --cache-dir <dataset_root>/meta/hlrp_action_frame_filter_cache \
  --split train \
  --all-files \
  --episode-row 0 \
  --fps 15
```

Rank episodes by filtered fraction (top-10, train split):

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

cache_dir = Path("cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache")
rows = []

for cache_path in sorted(cache_dir.glob("train_*.npz")):
  payload = np.load(cache_path, allow_pickle=False)
  episode_ids = payload["episode_ids"].astype(int)
  starts = payload["candidate_offsets_start"].astype(int)
  ends = payload["candidate_offsets_end"].astype(int)
  keep = payload["keep_mask"].astype(bool)
  for row_idx, episode_id in enumerate(episode_ids.tolist()):
    sl = slice(int(starts[row_idx]), int(ends[row_idx]))
    total = int(ends[row_idx] - starts[row_idx])
    kept = int(keep[sl].sum())
    dropped = total - kept
    drop_frac = float(dropped) / float(max(1, total))
    rows.append((drop_frac, dropped, total, int(episode_id), cache_path.name))

rows.sort(key=lambda x: x[0], reverse=True)
for rank, (drop_frac, dropped, total, episode_id, filename) in enumerate(rows[:10], start=1):
  print(
    f"{rank:2d}. drop_frac={drop_frac:.3f} dropped={dropped:4d}/{total:4d} "
    f"episode={episode_id:4d} file={filename}"
  )
PY

```

cache/huggingface/lerobot/LSY-lab/simple_tasks_human_rec_v0/meta/hlrp_action_frame_filter_cache
cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache

Rank episodes by absolute dropped anchors (top-10):

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

cache_dir = Path("cache/huggingface/lerobot/LSY-lab/simple_tasks_teleop_v1/meta/hlrp_action_frame_filter_cache")
rows = []

for cache_path in sorted(cache_dir.glob("train_*.npz")):
  payload = np.load(cache_path, allow_pickle=False)
  episode_ids = payload["episode_ids"].astype(int)
  starts = payload["candidate_offsets_start"].astype(int)
  ends = payload["candidate_offsets_end"].astype(int)
  keep = payload["keep_mask"].astype(bool)
  for row_idx, episode_id in enumerate(episode_ids.tolist()):
    sl = slice(int(starts[row_idx]), int(ends[row_idx]))
    total = int(ends[row_idx] - starts[row_idx])
    kept = int(keep[sl].sum())
    dropped = total - kept
    rows.append((dropped, total, int(episode_id), cache_path.name))

rows.sort(key=lambda x: x[0], reverse=True)
for rank, (dropped, total, episode_id, filename) in enumerate(rows[:10], start=1):
  print(f"{rank:2d}. dropped={dropped:4d}/{total:4d} episode={episode_id:4d} file={filename}")
PY
```

Check which cache file the current config actually uses:

```bash
conda run -n hlrp python scripts/2_train_stage1_lam.py \
  experiment=stage1_local \
  data=lsy_teleop_only \
  training.max_steps=1 \
  logging.use_wandb=false \
  data.adapter.lerobot_v3.steps_per_epoch=20 \
  2>&1 | rg "cache=hit|cache=score-hit|cache=miss|Action-frame filtering source|train_episodes|val_episodes"
```

Interpretation:
- `cache=hit` means the exact fingerprint file already exists and was reused.
- `cache=score-hit` means scores were reused and only keep/drop decisions were recomputed (for example threshold-only changes).
- `cache=miss` means no exact match existed, so a new fingerprinted file was generated.

Timing expectations (rough):
- `cache=hit`: filtering phase should usually be quick (often seconds to low tens of seconds).
- `cache=score-hit`: can still take noticeable time on large episode sets because decisions are recomputed across all candidate anchors.
- `cache=miss`: slowest path (rescoring/decoding), often minutes.

Dataset viz command pattern:

```bash
python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id LSY-lab/simple_tasks_teleop_v1 \
  --root cache/huggingface/lerobot \
  --display-compressed-images 0 \
  --mode local \
  --save 1 \
  --output-dir /tmp/lerobot_viz_simple_tasks_teleop_v1 \
  --episode-index 3

rerun /tmp/lerobot_viz_simple_tasks_teleop_v1/LSY-lab_simple_tasks_teleop_v1_episode_0.rrd

rerun /tmp/lerobot_viz_simple_tasks_human_v0/LSY-lab_simple_tasks_human_rec_v0_episode_0.rrd
```

Inspect a few episodes quickly:

```bash
for ep in 9 11 7 3; do
  python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id LSY-lab/simple_tasks_teleop_v1 \
    --root cache/huggingface/lerobot \
    --display-compressed-images 0 \
    --mode local \
    --save 1 \
    --output-dir /tmp/lerobot_viz_simple_tasks_teleop_v1 \
    --episode-index "${ep}"
done


for ep in 3; do
  python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id LSY-lab/simple_tasks_human_rec_v0 \
  --root cache/huggingface/lerobot/LSY-lab/simple_tasks_human_rec_v0 \
  --display-compressed-images False \
  --mode local \
  --save 1 \
  --output-dir /tmp/lerobot_viz_simple_tasks_human_v0 \
    --episode-index "${ep}"
done

rerun /tmp/lerobot_viz_simple_tasks_teleop_v1/simple_tasks_teleop_v1_episode_0.rrd
```

Notes:
- Your original command is valid; the improvements are using fully qualified repo id and the LeRobot cache root as `--root`.
- If your local root has a different folder layout, adjust `--root` only.
- Use fully qualified repo id (`LSY-lab/...`) to avoid resolution mismatch.

## Disable entirely

Set:

```yaml
data:
  filtering:
    enabled: false
    mode: none
```
