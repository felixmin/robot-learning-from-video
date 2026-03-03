# LeRobot v3 Dataset Unification Plan

Last updated: 2026-03-03

## Decision

Primary direction:

1. move Stage 1 and Stage 2 onto a shared HLRP-owned dataset contract,
2. use official LeRobot v3 datasets as the long-term source format,
3. build on top of standard `LeRobotDataset` for actual parquet/video access,
4. do not rely on `MultiLeRobotDataset` for mixing,
5. keep legacy raw OXE only as a temporary validation path until the LeRobot path is trusted.

This is a LeRobot-v3-first design, not a generic plugin system for arbitrary backends.

## Scope

This plan covers:

- Stage 1 data loading,
- Stage 2 data loading,
- weighted mixing across multiple LeRobot v3 datasets,
- distributed-sampling semantics for single-GPU, multi-GPU, and multi-node training,
- runtime/cache/resource behavior.

This plan does not yet replace Stage 3's current LeRobot train script. Stage 3 remains the parity target and later integration target, not the first code path to migrate.

## Current repo integration points

Current active code paths:

- `packages/common/data_factory.py`
  - only supports `data.backend=oxe_local_indexed`
- `packages/common/data.py`
  - `OpenXLocalDataModule`
  - `oxe_collate_fn`
- `packages/common/adapters/openx_local_indexed_full.py`
  - current indexed sampler and dataset
- `packages/laq/task.py`
  - Stage 1 batch contract today is `dict` with `frames` plus metadata
- `packages/foundation/vla_backend_module.py`
  - Stage 2 still adapts from OXE-shaped batches
- `packages/foundation/backends/interfaces.py`
  - `FoundationBatch` is already the correct internal contract for Stage 2 and Stage 3
- `lerobot/src/lerobot/datasets/lerobot_dataset.py`
  - standard LeRobot v3 data access layer
- `lerobot/src/lerobot/datasets/video_utils.py`
  - video decoder behavior and cache behavior

## Core architectural choice

Use this split:

1. HLRP request/sample contract
2. LeRobot-v3 source adapter
3. HLRP weighted mixer and sampler
4. Stage 1 batch adapter
5. thin Stage 2 adapter

This means:

- LeRobot handles file-format access, delta querying, and video decode.
- HLRP handles source weighting, episode/anchor sampling, and stage-specific batch adaptation.
- Stage 1 and Stage 2 share the same lower-level dataset boundary, but they do not need to share the same final stage-facing batch type.

## Why not use standard `MultiLeRobotDataset`

Vendored LeRobot `MultiLeRobotDataset` is not sufficient:

- it is concatenation-oriented, not weighted mixing,
- it keeps only the intersection of feature keys,
- it is effectively disabled in LeRobot training config/factory,
- it does not solve request-dependent anchor validity,
- it does not solve dataset-locality control.

## Proposed code layout

### New modules

Planned additions:

- `packages/common/lerobot_v3_types.py`
- `packages/common/lerobot_v3_source.py`
- `packages/common/lerobot_v3_sampler.py`
- `packages/common/lerobot_v3_data.py`
- `packages/common/lerobot_v3_adapters.py`
- `packages/common/lerobot_v3_stats.py`
- `packages/common/lerobot_v3_stats.py`

Planned updates:

- `packages/common/data_factory.py`
  - add backend switch for `lerobot_v3`
- `scripts/2_train_laq.py`
  - allow new backend
- `scripts/4_train_foundation.py`
  - allow new backend
- `packages/foundation/vla_backend_module.py`
  - replace OXE-only assumptions with canonical adapter input
- `packages/foundation/online_laq.py`
  - keep OXE helpers temporarily, add canonical-batch helpers

### Proposed top-level entrypoints

```python
# packages/common/data_factory.py
def create_datamodule(cfg_data: Any):
    if backend == "oxe_local_indexed":
        ...
    if backend == "lerobot_v3":
        from common.lerobot_v3_data import LeRobotV3DataModule
        return LeRobotV3DataModule(...)
```

## Canonical HLRP dataset contract

### Request object

```python
@dataclass(frozen=True)
class TemporalFieldRequest:
    deltas_steps: tuple[int, ...]
    required: bool = True


@dataclass(frozen=True)
class DatasetRequest:
    image_requests: dict[str, TemporalFieldRequest]
    state_request: TemporalFieldRequest | None = None
    action_request: TemporalFieldRequest | None = None
    include_task_text: bool = False
    include_subtask_text: bool = False
    include_metadata: bool = True
    pad_missing_future: bool = True
    image_size: tuple[int, int] | None = None
    image_dtype: str = "uint8"
```

Notes:

- stage requests use step offsets, not seconds,
- source converts step offsets to LeRobot `delta_timestamps` using source fps,
- camera keys in the request are HLRP-level canonical roles, e.g. `primary`, `wrist`.
- image resizing is part of source-side sample construction, not a later collate concern,
- source should return fixed-size resized `uint8` images when `image_size` is set,
- normalization remains stage/backend-specific and should not happen inside the dataset source.
- the request/sample contract is intentionally general enough for:
  - single frame pairs,
  - multi-camera frame pairs,
  - longer temporal image sequences such as `[t, t+X, t+2X, ...]`,
  - optional action/state windows,
  - optional task text.

### Sample object

```python
@dataclass
class DatasetSample:
    image_streams: dict[str, torch.Tensor] | None = None         # camera -> [T,C,H,W]
    image_padding_masks: dict[str, torch.Tensor] | None = None   # camera -> [T] bool
    state: torch.Tensor | None = None                            # [Ts,S]
    state_is_pad: torch.Tensor | None = None                     # [Ts] bool
    action: torch.Tensor | None = None                           # [Ta,A]
    action_is_pad: torch.Tensor | None = None                    # [Ta] bool
    task_text: str | None = None
    subtask_text: str | None = None
    meta: dict[str, Any] | None = None
```

### Batched sample object

```python
@dataclass
class BatchedDatasetSample:
    image_streams: dict[str, torch.Tensor] | None = None         # camera -> [B,T,C,H,W]
    image_padding_masks: dict[str, torch.Tensor] | None = None   # camera -> [B,T]
    state: torch.Tensor | None = None                            # [B,Ts,S]
    state_is_pad: torch.Tensor | None = None                     # [B,Ts]
    action: torch.Tensor | None = None                           # [B,Ta,A]
    action_is_pad: torch.Tensor | None = None                    # [B,Ta]
    task_text: list[str] | None = None
    subtask_text: list[str] | None = None
    meta: dict[str, Any] | None = None
```

### Stage-facing batch objects

Stage 1 and Stage 2 should both consume the same lower-level `BatchedDatasetSample`, but may adapt it into different final batch objects.

```python
@dataclass(frozen=True)
class Stage1Batch:
    image_streams: dict[str, torch.Tensor]                  # camera -> [B,T,C,H,W]
    image_padding_masks: dict[str, torch.Tensor] | None = None
    task_text: list[str] | None = None
    subtask_text: list[str] | None = None
    state: torch.Tensor | None = None                       # [B,Ts,S]
    state_is_pad: torch.Tensor | None = None                # [B,Ts]
    action: torch.Tensor | None = None                      # [B,Ta,A]
    action_is_pad: torch.Tensor | None = None               # [B,Ta]
    meta: dict[str, Any] | None = None
```

`FoundationBatch` remains the Stage-2 / Stage-3 internal batch type.

## Single source and mixed source share the same interface

```python
class DatasetSource(Protocol):
    def compile(self, request: DatasetRequest) -> None: ...
    def sample_token(self, rng: np.random.Generator) -> "SampleToken": ...
    def get_sample(self, token: "SampleToken") -> DatasetSample: ...
    def num_sampleable_episodes(self) -> int: ...
```

Two concrete implementations:

- `LeRobotSingleSource(DatasetSource)`
- `WeightedLeRobotMixer(DatasetSource)`

The mixer behaves like a single dataset from the stage's perspective.

## Anchor definition

`anchor` = the base timestep of a sample.

Examples:

- Stage 1 request: image pair at `[0, +5]`
  - anchor = timestep `t`
  - returned pair = `t` and `t+5`
- Stage 2 request: current image + current state + future action chunk `[0..+49]`
  - anchor = timestep `t`
  - returned sample = conditioning observations at `t`, supervision actions from `t` onward

Everything in the request is interpreted relative to the anchor.

## Metadata indexing design

### What is scanned

For each source, scan only:

- `meta/info.json`
- `meta/episodes/...`

Do not scan:

- full frame tables to enumerate all rows,
- video contents,
- all valid windows.

### Why metadata-only indexing

Because valid sampling regions depend on the request.

If we materialize all anchors:

- memory becomes `O(num_frames)` or worse,
- request changes require full rebuilds,
- weighted mixing over many datasets becomes more expensive to reset.

With metadata-only indexing:

- memory is `O(num_episodes)`,
- compiling a new request is cheap,
- sampling remains on-the-fly.

### Compiled source index

```python
@dataclass(frozen=True)
class CompiledEpisodeIndex:
    episode_index: np.ndarray          # [E] int32
    dataset_from_index: np.ndarray     # [E] int64
    dataset_to_index: np.ndarray       # [E] int64  (exclusive)
    valid_anchor_start: np.ndarray     # [E] int64
    valid_anchor_end: np.ndarray       # [E] int64  (exclusive)
    valid_anchor_count: np.ndarray     # [E] int32


@dataclass(frozen=True)
class CompiledSourceIndex:
    repo_id: str
    fps: int
    camera_role_to_key: dict[str, str]
    state_key: str | None
    action_key: str | None
    episodes: CompiledEpisodeIndex
    sampleable_episode_ids: np.ndarray
    sampleable_episode_weights: np.ndarray
```

### Functions

```python
def load_lerobot_meta(repo_id: str, root: str | Path | None, revision: str | None) -> LeRobotDatasetMetadata: ...

def resolve_request_to_delta_timestamps(
    request: DatasetRequest,
    *,
    fps: int,
    camera_role_to_key: dict[str, str],
    state_key: str | None,
    action_key: str | None,
) -> dict[str, list[float]]: ...

def compile_source_index(
    *,
    meta: LeRobotDatasetMetadata,
    request: DatasetRequest,
    camera_role_to_key: dict[str, str],
    state_key: str | None,
    action_key: str | None,
) -> CompiledSourceIndex: ...
```

### Valid-anchor rule

At compile time, derive:

- `min_delta_steps`
- `max_delta_steps`
- whether padding is allowed for future state/action windows

Then per episode:

- strict no-padding case:
  - `valid_anchor_start = ep_start - min_delta`
  - `valid_anchor_end = ep_end - max_delta`
- future-padding-allowed case:
  - action/state future requests may keep the full episode range
  - LeRobot `*_is_pad` handles the tail

For Stage 1 frame-pair training, the default should be strict valid pairs.

## Important LeRobot runtime constraint

Do not instantiate a separate `LeRobotDataset(..., episodes=[...])` per episode or per tiny subset.

Reason:

- `load_nested_dataset(..., episodes=...)` in LeRobot filters with PyArrow and builds a filtered table,
- this is materially heavier than keeping one map-style dataset object over the full local source,
- repeated filtered instantiation would add significant RAM and setup time.

So the source runtime should:

- keep one `LeRobotDataset` per source per worker,
- use full-dataset absolute frame indices for access,
- avoid per-episode sub-datasets.

## Runtime source behavior

### Worker-local runtime

```python
class LeRobotSourceRuntime:
    dataset: LeRobotDataset
    compiled_index: CompiledSourceIndex
    resolved_delta_timestamps: dict[str, list[float]]
```

### Source implementation

```python
class LeRobotSingleSource(DatasetSource):
    def compile(self, request: DatasetRequest) -> None: ...
    def sample_token(self, rng: np.random.Generator) -> SampleToken: ...
    def get_sample(self, token: SampleToken) -> DatasetSample: ...
    def _get_runtime(self) -> LeRobotSourceRuntime: ...
```

### Sample token

```python
@dataclass(frozen=True)
class SampleToken:
    source_id: int
    episode_id: int
    anchor_abs_index: int
```

Notes:

- `source_id` is an internal integer assigned from the resolved source list order during `LeRobotV3DataModule.setup()`,
- it only needs to be stable within one run,
- stable human-readable identity should always be carried separately in sample metadata, e.g. `source_name`, `repo_id`.

### Sampling policy inside one source

Default:

1. sample a valid episode from `sampleable_episode_ids`
2. sample an anchor uniformly from that episode's valid anchor range

This keeps memory low and matches the current OXE sampler's `dataset -> episode -> timestep` structure.

Optional future policies:

- uniform over anchors globally
- uniform over episodes
- temperature-adjusted episode weighting by length

## Mixer design

### Preferred training semantics

Training uses repeated weighted sources, not "one pass until source exhaustion".

That means:

- short datasets do not disappear mid-epoch,
- weights remain active for the full epoch,
- source episode orders cycle and reshuffle when exhausted.

### Mixer class

```python
class WeightedLeRobotMixer(DatasetSource):
    def compile(self, request: DatasetRequest) -> None: ...
    def sample_token(self, rng: np.random.Generator) -> SampleToken: ...
    def get_sample(self, token: SampleToken) -> DatasetSample: ...
```

### Weighted sampling rule

Default:

1. sample `source_id` from configured mixture weights
2. ask that source to sample one token
3. fetch that sample

Optional config:

- `weights_mode=explicit`
- `weights_mode=size_balanced`

`size_balanced` means:

- effective source mass = `source_weight * sampleable_anchor_mass`

Default recommendation for HLRP:

- `explicit`

because the user's research requirement is explicit, non-size-proportional mixing.

### Source cycling

Per source:

- keep a shuffled cycle of sampleable episodes,
- when exhausted, rebuild deterministically with `(seed, epoch, cycle_index)`,
- sample anchors on-the-fly within the chosen episode.

This is directly inspired by `OpenXLocalIndexedEpisodePairSampler`.

## Sampler and DataLoader contract

### Key decision

Do not build the full permutation of all valid anchors across all datasets.

Instead:

- compile source metadata once,
- sample tokens on-the-fly or build only an epoch-length token plan,
- let the dataset wrapper dispatch `token -> source -> DatasetSample`.

### Training epoch definition

Training epoch is a fixed number of sampled items, not a full pass over the underlying corpora.

Formula:

```text
global_samples_per_epoch = steps_per_epoch * batch_size * world_size
local_samples_per_epoch  = global_samples_per_epoch / world_size
```

If `steps_per_epoch` is not set, we may infer a default from weighted source mass, but explicit `steps_per_epoch` is preferred for mixed training.

### Sampler classes

```python
class WeightedLeRobotTokenSampler(torch.utils.data.Sampler[SampleToken]):
    def __init__(..., num_samples: int, seed: int, epoch: int, resample_each_epoch: bool): ...
    def __iter__(self) -> Iterator[SampleToken]: ...
    def __len__(self) -> int: ...
    def set_epoch(self, epoch: int) -> None: ...
```

Optional distributed variant:

```python
class DistributedWeightedLeRobotTokenSampler(WeightedLeRobotTokenSampler):
    def __init__(..., rank: int, world_size: int, global_num_samples: int): ...
```

### Dataset wrapper

```python
class LeRobotMixedMapDataset(torch.utils.data.Dataset):
    def __getitem__(self, token: SampleToken) -> DatasetSample: ...
    def __len__(self) -> int: ...
```

### Collate

```python
def collate_dataset_samples(batch: list[DatasetSample]) -> BatchedDatasetSample: ...
```

## Stage adapters

### Stage 1 adapter

Stage 1 should be upgraded to consume a dedicated `Stage1Batch`.

This is preferable to forcing richer temporal/multi-camera requests back into a legacy single-`frames` tensor layout.

Planned adapter:

```python
def dataset_batch_to_stage1_batch(
    batch: BatchedDatasetSample,
    *,
    camera_roles: tuple[str, ...] | None = None,
) -> Stage1Batch: ...
```

Expected output:

- `image_streams`: camera-role keyed temporal tensors
- `image_padding_masks`
- optional `task_text`
- optional `state`
- optional `action`
- `meta`

Recommended Stage-1 consumer update:

- migrate `packages/laq/task.py` and Stage-1-adjacent utilities from assuming a single `frames` tensor
- allow Stage 1 to explicitly choose:
  - one camera pair,
  - multiple camera pairs,
  - one or more temporal image sequences,
  - optional language/state/action conditioning

Backward compatibility option:

- if needed for a short transition, keep a compatibility helper:

```python
def stage1_batch_to_legacy_laq_dict(
    batch: Stage1Batch,
    *,
    camera_role: str,
) -> dict[str, Any]: ...
```

but this should not be the target architecture.

### Stage 2 adapter

Planned adapter:

```python
def dataset_batch_to_foundation_batch(batch: BatchedDatasetSample) -> FoundationBatch: ...
```

This should replace the OXE-only extraction path currently embedded in `packages/foundation/vla_backend_module.py`.

### Stage 3 parity adapter

No immediate code change planned.

Instead:

- keep Stage 3 as the reference LeRobot-native path,
- write parity tests comparing Stage 2's `FoundationBatch` construction against Stage 3's expectations.

Current limitation to keep in mind:

- LeRobot and the HLRP LeRobot policy config can already request two observation steps for all observation keys,
- but the current online-LAQ target-generation code in Stage 3 still assumes exactly one LAQ camera and exactly two observation steps.

## DataModule design

```python
class LeRobotV3DataModule(pl.LightningDataModule):
    def setup(self, stage: str | None = None) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
```

Responsibilities:

- resolve configured dataset specs,
- compile source indices,
- create train/val samplers,
- create single-source or mixed dataset wrapper,
- expose one collate function.

Suggested config surface:

```yaml
data:
  backend: lerobot_v3
  request:
    stage1:
      latent_camera_role: primary
      image_deltas_steps: [0, 5]
      image_size: [224, 224]
    stage2:
      camera_roles: [primary, wrist]
      image_deltas_steps: [0]
      image_size: [224, 224]
      state_deltas_steps: [0]
      action_horizon_steps: 50
      include_task_text: true
  dataset:
    lerobot:
      sources:
        - repo_id: lerobot/jaco_play
          weight: 1.0
          camera_map: {primary: observation.images.image, wrist: observation.images.image_wrist}
        - repo_id: lerobot/taco_play
          weight: 1.0
          camera_map: {primary: observation.images.rgb_static, wrist: observation.images.rgb_gripper}
  loader:
    batch_size: 64
    num_workers: 8
    pin_memory: true
    prefetch_factor: 1
    persistent_workers: true
  mixer:
    steps_per_epoch: 1000
    resample_each_epoch: true
    weights_mode: explicit
    seed: 42
```

## Caching and storage behavior

### Disk-level caches

1. Hugging Face / LeRobot local dataset cache
   - stores downloaded parquet/mp4/meta files
2. OS page cache
   - caches recently accessed parquet and video file pages
3. HLRP metadata-index cache
   - planned on-disk cache for compiled episode metadata
   - e.g. `cache/lerobot_v3_index/<repo_hash>/<request_hash>.npz`

### Process-level caches

1. `LeRobotDataset.hf_dataset`
   - one per source per worker
2. `LeRobotDataset.meta`
   - one per source per worker
3. `torchcodec` decoder cache in LeRobot
   - one decoder per `video_path` per worker process

### Planned HLRP worker cache

```python
class WorkerSourceRuntimeCache:
    def get(self, source_id: int) -> LeRobotSourceRuntime: ...
```

Policy:

- no eviction in v1,
- one lazy runtime per source per worker,
- keep runtimes alive across batches within persistent workers,
- revisit only if file-descriptor pressure or per-worker RAM becomes a measured problem.

### Why worker-local cache matters

Without it:

- every sample would rebuild `LeRobotDataset` objects,
- video decoder caches would never warm,
- metadata setup cost would repeat unnecessarily.

Practical expectation:

- this scales linearly with the number of active sources per worker,
- it is simple and appropriate for small to moderate mixtures,
- if a future run mixes many dozens of datasets, cache pressure becomes a profiling issue, not a v1 design requirement.

## Expected RAM overhead

The dominant cost is not metadata. It is in-flight decoded image/video tensors.

### Persistent metadata RAM

Per source:

- `CompiledSourceIndex` stores around 6 integer arrays over episodes
- rough lower bound:
  - `6 arrays * 8 bytes * num_episodes`

Examples:

- `10,000` episodes:
  - about `0.48 MB` raw array storage
- `100,000` episodes:
  - about `4.8 MB` raw array storage

Real Python object overhead will raise this somewhat, but it remains small relative to image tensors.

### Token-plan RAM

If we materialize only the epoch token plan:

- store `source_id`, `episode_id`, `anchor_abs_index`
- approximately `16-24 bytes` per token depending on dtype/alignment

Examples:

- `100,000` samples per epoch:
  - about `1.6-2.4 MB`
- `1,000,000` samples per epoch:
  - about `16-24 MB`

This is acceptable. It is still far smaller than materializing all anchors.

### Per-sample tensor RAM

Approximate image payload only:

- `1 camera`, `2 frames`, `224x224`, `uint8`
  - `2 * 224 * 224 * 3 = 301,056 bytes`
  - about `0.29 MB`
- same sample in `float32`
  - about `1.15 MB`
- `2 cameras`, `2 frames`, `512x512`, `float32`
  - about `12.0 MB`

State/action are negligible compared to images.

### In-flight DataLoader RAM

Approximate upper bound:

```text
num_workers * prefetch_factor * batch_size * bytes_per_sample
```

Example:

- `num_workers=8`
- `prefetch_factor=1`
- `batch_size=32`
- `bytes_per_sample=1.15 MB`

Upper-bound image payload in flight:

- about `294 MB`

At higher resolution or multi-camera float batches, this grows quickly.

### RAM recommendations

- keep worker-side images as `uint8` if possible,
- use `prefetch_factor=1` initially,
- use `persistent_workers=True`,
- do not instantiate per-episode filtered `LeRobotDataset`s.

## Expected CPU overhead

Main CPU costs:

1. metadata compile
2. video decode
3. deterministic image resize
4. Python dispatch and collation

### Metadata compile time

Expected one-time per source:

- `meta/info.json`
  - negligible
- `meta/episodes/...`
  - local SSD: typically sub-second to a few seconds
  - networked storage: can be several seconds

Expected order of magnitude:

- `0.2-3 s` per source local
- `2-10+ s` per source on slower remote storage

This is a setup cost, not per-batch cost.

### Video decode cost

This is the main runtime CPU bottleneck.

Dominated by:

- number of cameras
- number of queried frames
- frame resolution
- backend (`torchcodec` vs `pyav`)
- locality of access to the same mp4 path

Expected relative behavior:

- `torchcodec` with warm decoder cache is best
- `pyav` / `video_reader` reopen readers more often and cost more CPU per query
- random source/file switching hurts cache reuse

### Resize cost

Resize should happen in the source adapter immediately after image decode and before collation.

Reason:

- fixed-size tensors make batching simple,
- this matches the current OXE path more closely,
- leaving images at native resolution would force variable-shape handling downstream.

Expected cost:

- lower than video decode but still meaningful,
- grows with number of cameras and temporal frames,
- should stay on CPU in the source path; later model-specific normalization can remain in the backend.

## Expected bottlenecks

### 1. Video decode and image transforms

Most likely runtime bottleneck for Stage 1 and Stage 2.

Why:

- LeRobot already handles padding/query logic efficiently enough
- images dominate bytes and CPU
- multi-camera future windows multiply decode work

Mitigation:

- use `torchcodec` when available
- keep `persistent_workers=True`
- start with `prefetch_factor=1`

### 2. Random source/file switching

Why:

- source switches reduce reuse of warm dataset objects and warm video decoders
- each worker keeps its own decoder cache

Mitigation:

- accept this in v1 and profile before adding locality knobs
- if it becomes a real bottleneck, revisit sampler-level locality as an optimization pass, not initial architecture

### 3. Remote downloads / cache misses

Worst-case bottleneck.

If files are not already local:

- first access can stall workers by seconds or longer,
- mixture training amplifies this by touching more sources.

Mitigation:

- pre-stage chosen datasets locally before training,
- use metadata-only probes during validation,
- keep runtime Hub downloads out of the hot path.

### 4. Distributed data imbalance

If rank-local batch counts differ:

- training can hang or waste GPU time waiting at sync points.

Mitigation:

- sampler must guarantee equal local sample counts per rank,
- training epochs must be fixed-length, not exhaustion-based.

## Distributed semantics

### Training

Preferred design:

- sampler owns deterministic epoch semantics,
- rank-local sample counts are equal,
- short datasets cycle and reshuffle instead of disappearing,
- configured weights remain active for the full epoch.

### Stage 1 and Stage 2

Planned approach:

- HLRP-owned sampler,
- disable framework auto-replacement of the sampler,
- explicit `set_epoch(epoch)` support.

### Stage 3

Not first implementation target.

When integrating later:

- either use rank-local dataloaders directly,
- or ensure Accelerate sharding does not distort mixer semantics.

## Similarities and differences vs standard LeRobot

### Same as standard LeRobot

- same dataset format
- same metadata files
- same parquet/video access layer
- same `delta_timestamps`
- same `*_is_pad`
- same multi-camera support

### Added by HLRP

- explicit weighted source mixing
- request-aware valid-anchor compilation
- source/episode/anchor sampling
- same source interface for single and mixed datasets
- stage-specific canonical adapters

## Validation split semantics

Validation should be finite and deterministic, not an infinite weighted mixture.

Design:

- each source config should define how train and validation episodes are selected,
- `LeRobotV3DataModule.setup()` should compile separate train and validation indices,
- validation should use a deterministic finite sampler over held-out episodes or held-out anchors,
- validation metadata should always include `source_name` and `dataset_name` so existing bucket-aware reporting remains usable.

Initial recommendation:

- keep train/val source lists the same unless config says otherwise,
- split by held-out episodes per source,
- do not oversample small validation sets just to mimic training weights.

This keeps validation interpretable and avoids infinite-loop semantics in evaluation.

## Action/state semantics across mixed datasets

The dataset source should return native action/state tensors plus padding masks.

That means:

- no cross-source action normalization inside the dataset source,
- no forced action projection inside the dataset source,
- sample metadata should include native dimensions where needed, e.g. `action_dim`, `state_dim`.

Stage-2 adapter responsibilities:

- pad to model-required dimensions,
- apply normalization using the chosen training-time statistics,
- pass a single shared normalization object into the backend/model path.

Chosen v1 policy:

- assume the configured mixed sources are action/state compatible,
- use one global normalization space per run,
- do not use per-dataset normalization,
- keep raw canonical action/state tensors in the dataset interface until adapter/backend normalization.

How normalization stats are obtained:

- single-source run:
  - use that source's `meta.stats` directly
- mixed run:
  - compute one weighted merged stats object from all selected sources' `meta.stats`
  - weights should match the configured mixture weights rather than raw dataset sizes alone

Reason:

- this matches current Stage 2 and Stage 3 code structure best,
- Stage 2 already expects one `normalization_stats` object,
- Stage 3 HLRP policy already expects one `dataset_stats` bundle,
- per-dataset normalization would make normalized targets source-dependent and complicate both training and inference.

Planned helper surface:

```python
def load_source_stats(meta: LeRobotDatasetMetadata) -> dict[str, dict[str, np.ndarray]]: ...

def merge_weighted_stats(
    stats_by_source: list[dict[str, dict[str, np.ndarray]]],
    source_weights: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]: ...

def build_run_normalization_stats(
    sources: Sequence["LeRobotSingleSource"],
    *,
    weights_mode: str,
) -> dict[str, dict[str, np.ndarray]]: ...
```

Implementation note:

- `merge_weighted_stats(...)` should follow LeRobot's global aggregation logic for `mean/std/min/max`,
- but use the configured mixture weights to scale each source contribution,
- this is the stats object that should be passed into Stage 2 and Stage 3 code paths.

## Validation and tests

### Test layout

Keep correctness tests in `tests/` and throughput experiments in `experiments/benchmarking/`.

Planned new test files:

- `tests/common/lerobot_v3_fixtures.py`
- `tests/test_lerobot_v3_request_resolution.py`
- `tests/test_lerobot_v3_stats.py`
- `tests/test_lerobot_v3_index_compile.py`
- `tests/test_lerobot_v3_sampler.py`
- `tests/test_lerobot_v3_collate.py`
- `tests/test_lerobot_v3_stage1_adapter.py`
- `tests/test_lerobot_v3_foundation_adapter.py`
- `tests/test_lerobot_v3_datamodule.py`
- `tests/test_lerobot_v3_smoke.py`

### Shared pytest helpers

Put synthetic builders in `tests/common/lerobot_v3_fixtures.py`.

Planned helper surface:

```python
def make_test_request(...) -> DatasetRequest: ...
def make_test_meta(...) -> LeRobotDatasetMetadata: ...
def make_test_source_stats(...) -> dict[str, dict[str, np.ndarray]]: ...
def make_compiled_source_index(...) -> CompiledSourceIndex: ...
def make_dataset_sample(...) -> DatasetSample: ...
def make_batched_dataset_sample(...) -> BatchedDatasetSample: ...
def make_sample_tokens(...) -> list[SampleToken]: ...
```

Goals:

- unit tests should not depend on network or real downloaded datasets,
- synthetic tests should exercise the contract and sampler deterministically,
- live dataset smoke tests should be opt-in and small.

### Unit tests

Planned tests:

- request-to-delta-timestamp resolution
- valid-anchor range compilation
- weighted source proportions over many draws
- episode-cycle rebuild behavior
- determinism for same `seed` and `epoch`
- different sequence after `set_epoch`
- equal per-rank lengths for distributed sampler

Detailed breakdown:

#### `tests/test_lerobot_v3_request_resolution.py`

Focus:

- request -> LeRobot `delta_timestamps` compilation
- camera role mapping
- image/state/action key selection

Planned test functions:

```python
def test_resolve_request_to_delta_timestamps_maps_camera_roles() -> None: ...
def test_resolve_request_to_delta_timestamps_converts_steps_using_fps() -> None: ...
def test_resolve_request_to_delta_timestamps_omits_unrequested_modalities() -> None: ...
def test_resolve_request_to_delta_timestamps_rejects_missing_required_camera_role() -> None: ...
```

#### `tests/test_lerobot_v3_stats.py`

Focus:

- single-source stats passthrough
- weighted mixed-stats aggregation
- parity with LeRobot aggregation when all weights are equal

Planned test functions:

```python
def test_merge_weighted_stats_returns_single_source_stats_unchanged() -> None: ...
def test_merge_weighted_stats_respects_source_weights_for_mean_and_std() -> None: ...
def test_merge_weighted_stats_keeps_global_min_and_max() -> None: ...
def test_merge_weighted_stats_matches_lerobot_aggregate_stats_for_equal_weights() -> None: ...
def test_build_run_normalization_stats_uses_selected_sources_only() -> None: ...
```

#### `tests/test_lerobot_v3_index_compile.py`

Focus:

- metadata-only index compilation
- valid-anchor bounds
- strict-pair vs future-padding behavior

Planned test functions:

```python
def test_compile_source_index_computes_valid_anchor_ranges_for_strict_pair_request() -> None: ...
def test_compile_source_index_allows_full_episode_range_when_future_padding_enabled() -> None: ...
def test_compile_source_index_drops_unsampleable_episodes() -> None: ...
def test_compile_source_index_uses_absolute_dataset_indices() -> None: ...
```

#### `tests/test_lerobot_v3_sampler.py`

Focus:

- weighted dataset sampling
- episode-cycle behavior
- deterministic epoch reseeding
- distributed partition invariants

Planned test functions:

```python
def test_weighted_lerobot_token_sampler_respects_configured_source_weights() -> None: ...
def test_weighted_lerobot_token_sampler_rebuilds_episode_cycle_after_exhaustion() -> None: ...
def test_weighted_lerobot_token_sampler_is_deterministic_for_same_seed_and_epoch() -> None: ...
def test_weighted_lerobot_token_sampler_set_epoch_changes_sequence() -> None: ...
def test_distributed_weighted_lerobot_token_sampler_produces_equal_rank_lengths() -> None: ...
def test_distributed_weighted_lerobot_token_sampler_has_no_cross_rank_token_overlap_for_epoch_plan() -> None: ...
```

This should follow the style already used in [test_openx_indexed_sampler.py](/mnt/data/workspace/code/high-level-robot-planner/tests/test_openx_indexed_sampler.py).

#### `tests/test_lerobot_v3_collate.py`

Focus:

- canonical batched sample construction
- optional field handling
- metadata preservation

Planned test functions:

```python
def test_collate_dataset_samples_stacks_multicamera_temporal_tensors() -> None: ...
def test_collate_dataset_samples_handles_missing_optional_fields() -> None: ...
def test_collate_dataset_samples_preserves_string_lists_and_meta_lists() -> None: ...
def test_collate_dataset_samples_rejects_mismatched_camera_sets() -> None: ...
```

#### `tests/test_lerobot_v3_stage1_adapter.py`

Focus:

- `BatchedDatasetSample -> Stage1Batch`
- pair and sequence use cases
- multi-camera Stage 1 path

Planned test functions:

```python
def test_dataset_batch_to_stage1_batch_preserves_multicamera_temporal_shape() -> None: ...
def test_dataset_batch_to_stage1_batch_handles_optional_action_and_state() -> None: ...
def test_stage1_batch_to_legacy_laq_dict_extracts_single_camera_pair() -> None: ...
def test_stage1_batch_to_legacy_laq_dict_rejects_non_pair_temporal_length() -> None: ...
```

#### `tests/test_lerobot_v3_foundation_adapter.py`

Focus:

- `BatchedDatasetSample -> FoundationBatch`
- normalization/padding boundary
- Stage 2 / Stage 3 parity at canonical batch level

Planned test functions:

```python
def test_dataset_batch_to_foundation_batch_maps_core_fields() -> None: ...
def test_dataset_batch_to_foundation_batch_preserves_action_is_pad() -> None: ...
def test_dataset_batch_to_foundation_batch_keeps_raw_actions_before_backend_normalization() -> None: ...
def test_dataset_batch_to_foundation_batch_matches_stage3_core_field_expectations() -> None: ...
```

This should complement [test_stage2_stage3_transform_parity.py](/mnt/data/workspace/code/high-level-robot-planner/tests/test_stage2_stage3_transform_parity.py).

#### `tests/test_lerobot_v3_datamodule.py`

Focus:

- backend wiring
- train/val separation
- single-source vs mixed-source uniform interface

Planned test functions:

```python
def test_lerobot_v3_datamodule_builds_single_source_dataset() -> None: ...
def test_lerobot_v3_datamodule_builds_weighted_mixed_dataset() -> None: ...
def test_lerobot_v3_datamodule_builds_separate_train_and_val_indices() -> None: ...
def test_create_datamodule_supports_lerobot_v3_backend() -> None: ...
```

### Live smoke tests

Keep live dataset tests separate and opt-in because they may download hundreds of MB.

File:

- `tests/test_lerobot_v3_smoke.py`

Planned behavior:

- skip unless an env var such as `HLRP_RUN_LEROBOT_SMOKE=1` is set,
- use one or two tiny representative sources already downloaded locally if available,
- otherwise allow a bounded small download into a temporary cache.

Planned test functions:

```python
def test_lerobot_single_source_runtime_smoke_jaco_play() -> None: ...
def test_lerobot_single_source_runtime_smoke_taco_play() -> None: ...
def test_weighted_mixer_runtime_smoke_two_sources() -> None: ...
```

Each smoke test should verify:

- sample retrieval succeeds,
- requested camera/time fields are present,
- pad masks are the expected rank,
- Stage 1 and Stage 2 adapters can consume the batch.

### Integration tests

- local smoke on `lerobot/jaco_play`
- local smoke on `lerobot/taco_play`
- mixed-source smoke on two datasets
- Stage 1 batch-shape parity against current LAQ expectations
- Stage 2 `FoundationBatch` parity against current backend expectations

### Semantic parity checks before removing legacy OXE

For representative datasets:

1. task text preserved
2. camera mapping correct
3. fps / temporal deltas correct
4. action chunk semantics correct
5. padding behavior correct
6. chosen canonical `observation.state` acceptable

Quantitative parity recommendation:

- for each representative dataset, sample a fixed set of anchors from both paths,
- compare metadata alignment exactly,
- compare padding masks exactly,
- compare resized image tensors with a tolerant image-difference metric after matching decode/resize settings,
- compare action/state tensor shapes and values up to the expected canonicalization differences,
- record pass/fail counts and summary statistics in `docs/`.

## Benchmark plan

Use `experiments/benchmarking/`, matching the existing benchmark layout.

Planned new file:

- `experiments/benchmarking/bench_lerobot_v3_dataloader.py`

Purpose:

- measure pure dataloader throughput for the new LeRobot-v3 backend without model forward/backward cost,
- compare single-source and mixed-source behavior,
- sweep batch size and worker count,
- record setup cost separately from steady-state throughput.

### Benchmark script structure

Planned top-level functions:

```python
def _percentile(values: list[float], q: float) -> float: ...
def _estimate_sample_payload_bytes(batch: BatchedDatasetSample) -> int: ...
def _summarize_batch(batch: BatchedDatasetSample) -> dict[str, Any]: ...
def _measure_loader(
    loader: DataLoader,
    *,
    warmup_steps: int,
    measured_steps: int,
    compute_sleep_s: float,
) -> dict[str, Any]: ...
def run_single_benchmark(cfg: DictConfig) -> dict[str, Any]: ...
def run_benchmark_grid(cfg: DictConfig) -> list[dict[str, Any]]: ...
def write_benchmark_results(results: list[dict[str, Any]], output_dir: Path) -> None: ...
```

Hydra entrypoint:

```python
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None: ...
```

### What the benchmark should sweep

Primary sweep dimensions:

- `data.loader.batch_size`
- `data.loader.num_workers`
- `data.loader.prefetch_factor`
- number of mixed sources
- requested image size
- requested cameras / temporal length

Minimum recommended default grid:

```yaml
benchmark:
  warmup_steps: 20
  measured_steps: 100
  compute_sleep_s: 0.0
  batch_sizes: [8, 16, 32, 64]
  num_workers_list: [0, 2, 4, 8]
  prefetch_factors: [1]
  source_setups: ["single", "mixed2"]
```

### Metrics to record

For each run:

- `setup_time_s`
- `first_batch_time_s`
- `mean_batch_time_s`
- `p50_batch_time_s`
- `p90_batch_time_s`
- `p99_batch_time_s`
- `batches_per_s`
- `samples_per_s`
- `approx_batch_payload_mb`
- `rss_mb_before`
- `rss_mb_after`
- `batch_size`
- `num_workers`
- `prefetch_factor`
- `num_sources`
- `camera_roles`
- `image_size`
- `image_time_steps`
- `action_horizon_steps`

Optional if easy to wire:

- decoder backend in use (`torchcodec` vs other),
- per-run cache root,
- whether source is local-only or network-backed.

### Output format

Write benchmark outputs into the Hydra run dir:

- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_summary.txt`

This should mirror the style of existing scripts in `experiments/benchmarking/`.

### Benchmark scenarios to include

At minimum:

1. single source, one camera, pair request
2. single source, two cameras, pair request
3. single source, one camera, action chunk request
4. two-source weighted mix, one camera, pair request
5. two-source weighted mix, Stage-2-like request

Purpose of each:

- isolate camera-count scaling,
- isolate action-window cost,
- quantify source-mixing overhead,
- find the batch-size knee where throughput stops improving.

### Bottleneck diagnosis output

The benchmark script should log enough context to explain throughput regressions:

- whether workers are persistent,
- which batch fields are populated,
- approximate tensor payload size,
- whether first-batch cost dominates setup,
- whether throughput degrades when going from single-source to mixed-source.

## Test execution order

Recommended implementation/test order:

1. request/stats/index unit tests
2. sampler tests
3. collate + adapter tests
4. datamodule construction tests
5. live smoke tests
6. throughput benchmark runs

This keeps failures local and makes profiling happen only after correctness is established.

## Open questions

1. exact canonical camera-role set
   - likely `primary`, `wrist`, optional extras
2. exact weighting semantics
   - explicit only vs optional size-balanced mode
3. whether to implement a temporary legacy-OXE source adapter in code or only keep validation scripts
4. Stage 3 integration timing
   - defer until Stage 1/2 migration is stable

## Implementation phases

### Phase 1: contract and index

- add `DatasetRequest`, `DatasetSample`, `BatchedDatasetSample`, `SampleToken`
- add metadata-compile utilities
- add source runtime wrapper

### Phase 2: single-source LeRobot backend

- add `LeRobotSingleSource`
- add `LeRobotMixedMapDataset`
- add `collate_dataset_samples`
- add `LeRobotV3DataModule`

### Phase 3: weighted mixer

- add `WeightedLeRobotMixer`
- add `WeightedLeRobotTokenSampler`
- add distributed-aware epoch semantics
- add weighted global normalization-stats merge for the configured source mix

### Phase 4: stage adapters

- add `Stage1Batch`
- add `dataset_batch_to_stage1_batch`
- add `dataset_batch_to_foundation_batch`
- update Stage 1 consumer code to read `Stage1Batch`
- wire Stage 1 and Stage 2 to the new backend

### Phase 5: parity validation

- compare against legacy OXE on 2-3 datasets
- record quantitative parity metrics in `docs/`
- run Stage 1 and Stage 2 smoke training
- decide whether legacy OXE loader can be retired
