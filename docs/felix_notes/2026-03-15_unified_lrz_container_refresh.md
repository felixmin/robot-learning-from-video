# Unified LRZ Container Refresh

Date: 2026-03-15

## Summary

Validated the unified LRZ container deployment path and updated the helper
workflow toward:

1. build `containers/Dockerfile.unified`
2. push `felixmin/hlrp:unified-cuda-cu128`
3. import a dated `.sqsh` under DSS `enroot/`
4. archive the previously configured image into `enroot/old/`
5. update `config/user_config/local.yaml`

## Live Blockers Hit

- The LRZ refresh skill/scripts still assumed split `stage12` / `stage3`
  images, so unified deploys required manual overrides.
- `docker push` for the unified image is slow enough that long repeated
  `Waiting` output must not be treated as failure.
- Enroot import failed when GNU `parallel` wrote scratch buffers to home tmp:
  - `Cannot append to buffer file in /dss/dsshome1/.../tmp`
- Enroot import failed again when Enroot cache still targeted home:
  - `mktemp: failed to create file via template '/dss/dsshome1/.../enroot/cache/...`
- The repo still documented unification as future work in some operator docs.
- There was no scripted archive-and-activate step for unified images.

## Validated Fix

- Keep Enroot import scratch/cache on DSS by default:
  - `TMPDIR`
  - `PARALLEL_TMPDIR`
  - `ENROOT_CACHE_PATH`
- Treat `unified` as the canonical LRZ deployment profile.
- Make activation config-driven instead of renaming a new image into a fixed
  stage-specific target path.

## Provenance

- Docker tag: `felixmin/hlrp:unified-cuda-cu128`
- Pushed digest: `sha256:89c920ecd8a0bc69ee58522731f26907c052d49468967b6806f8c13778c1c3f7`
- Successful import job id: `5519164`
- Imported image:
  `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_unified_cu128_imported_2026-03-15_160458.sqsh`
- Current default operator config target:
  `config/user_config/local.yaml`
