# LRZ Docker Enroot Refresh Workflow

## Defaults

- Local repo root: `/mnt/data/workspace/code/high-level-robot-planner`
- Docker build context: `/mnt/data/workspace/code/high-level-robot-planner`
- Cluster host: `ai`
- Cluster Enroot dir: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot`
- Archive dir: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/old`
- Default activation target config: `/mnt/data/workspace/code/high-level-robot-planner/config/user_config/local.yaml`

## Canonical Unified Profile

- Dockerfile: `/mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.unified`
- Pushed tag: `felixmin/hlrp:unified-cuda-cu128`
- Imported image path shape:
  `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_unified_cu128_imported_<timestamp>.sqsh`

## Validated Command Shapes

Local build/push:

```bash
bash .codex/skills/lrz-docker-enroot-refresh/scripts/build_push_prune.sh \
  --profile unified
```

Notes:

- `docker push` can sit in repeated `Waiting` state for a long time on large layers.
- Do not treat long upload waits as failure unless Docker exits non-zero.

Cluster import:

```bash
bash .codex/skills/lrz-docker-enroot-refresh/scripts/submit_enroot_import.sh \
  --profile unified
```

This keeps all import scratch/cache paths on DSS by default:

- `TMPDIR=<enroot-dir>/tmp`
- `PARALLEL_TMPDIR=<enroot-dir>/tmp`
- `ENROOT_CACHE_PATH=<enroot-dir>/cache`

Activation/archive:

```bash
bash .codex/skills/lrz-docker-enroot-refresh/scripts/swap_enroot_image.sh \
  --replacement /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_unified_cu128_imported_<timestamp>.sqsh \
  --docker-tag felixmin/hlrp:unified-cuda-cu128 \
  --digest <pushed-digest> \
  --import-job-id <jobid>
```

Activation behavior:

- reads the currently configured image from `config/user_config/local.yaml` by default
- moves the old `.sqsh` into `enroot/old/`
- leaves the new `.sqsh` in `enroot/`
- updates the target config file
- appends provenance to `enroot/old/activation_log.tsv`

## Known Failure Modes

- If helper scripts still require a `stage12` override for unified images, the repo is out of sync and the workflow should be updated before the next deployment.
- `enroot import` can fail if GNU `parallel` writes buffers to home tmp:
  - `Cannot append to buffer file in /dss/dsshome1/.../tmp`
  - validated fix: keep `TMPDIR` and `PARALLEL_TMPDIR` on DSS
- `enroot import` can fail if Enroot cache still points to home:
  - `mktemp: failed to create file via template '/dss/dsshome1/.../enroot/cache/...`
  - validated fix: keep `ENROOT_CACHE_PATH` on DSS
- `enroot import` can OOM while creating squashfs.
  - validated fix remains `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`
- Activation should not move the currently configured image while jobs are queued or running unless the operator explicitly overrides the guard.
- `enroot/old/` will grow over time because deletion is forbidden by cluster policy.

## Provenance Expectations

For each activation, preserve:

- previous active image path
- archived image path
- new active image path
- Docker tag
- pushed digest
- import job id

The activation helper writes these fields into `enroot/old/activation_log.tsv`.
