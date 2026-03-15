# HLRP Containers

The unified container is now the intended runtime for all stages:

- `containers/Dockerfile.unified`
  Builds one raw-CUDA image with a single Python 3.10 environment at
  `/opt/hlrp-venv` for stage 1, stage 2, and stage 3.
- `containers/requirements.unified.txt`
  Holds the shared Python dependency set, including local installs of
  `lerobot` and `lerobot_policy_hlrp`.

The torch stack is installed explicitly in the Dockerfile so the wheel channel
can stay parameterized by build args. The currently validated unified image uses
`PYTORCH_WHL_CHANNEL=cu128` on both cluster H100 and local RTX 5090 targets.

## Cluster Setup

Cluster presets no longer hardcode a user-specific container path. The normal
operator workflow keeps a dated imported `.sqsh` under DSS `enroot/`, archives
the previously active image into `enroot/old/`, and updates
`config/user_config/local.yaml` to the new path.

The default operator target config is `config/user_config/local.yaml`:

```yaml
cluster:
  container:
    image: /dss/.../enroot/hlrp_unified_cu128_imported_2026-03-14_2248.sqsh
```

The shared cluster configs already set:

```yaml
cluster:
  container:
    python_bin: /opt/hlrp-venv/bin/python
```

so stage 1, stage 2, and stage 3 all use the same interpreter inside the
unified image.

## Refresh Workflow

1. Build on the workstation.
2. Push the image tag to Docker Hub.
3. Import that tag to the cluster with Enroot.
4. Keep Enroot scratch and cache on DSS, not home.
5. Archive the previously configured image into `enroot/old/`.
6. Update `config/user_config/local.yaml` to the new imported `.sqsh`.

Example build command:

```bash
bash .codex/skills/lrz-docker-enroot-refresh/scripts/build_push_prune.sh \
  --profile unified
```

Example Enroot import target:

```bash
/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/<user>/enroot/hlrp_unified_cu128_imported_<date>.sqsh
```

The validated import helper keeps its scratch/cache on DSS by default:

- `TMPDIR=<enroot-dir>/tmp`
- `PARALLEL_TMPDIR=<enroot-dir>/tmp`
- `ENROOT_CACHE_PATH=<enroot-dir>/cache`

Activation is handled by:

```bash
bash .codex/skills/lrz-docker-enroot-refresh/scripts/swap_enroot_image.sh \
  --replacement /dss/.../enroot/hlrp_unified_cu128_imported_<date>.sqsh \
  --docker-tag felixmin/hlrp:unified-cuda-cu128 \
  --digest <pushed-digest> \
  --import-job-id <jobid>
```

This keeps the new image in `enroot/`, moves the previously configured image
into `enroot/old/`, and records provenance in `enroot/old/activation_log.tsv`.
