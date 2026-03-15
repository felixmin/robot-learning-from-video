---
name: lrz-docker-enroot-refresh
description: "Build and refresh the unified LRZ container workflow end-to-end: build and push `containers/Dockerfile.unified` as `felixmin/hlrp:unified-cuda-cu128`, import it to DSS as a dated Enroot `.sqsh`, archive the previously configured image into `enroot/old/`, and update `config/user_config/local.yaml`."
---

# LRZ Docker Enroot Refresh

Use this workflow when refreshing the unified cluster container from the workstation. Keep local Docker steps on the workstation, keep cluster work behind `ssh ai`, never delete cluster images, keep Enroot scratch/cache paths on DSS, and archive old images into `enroot/old/` rather than deleting them.

Read [workflow.md](references/workflow.md) for the defaults, paths, and failure modes. Use the scripts in `scripts/` rather than rewriting the commands.

## Workflow

1. Verify execution context with `pwd` and `hostname`.
2. Use the canonical unified profile:
   - Dockerfile: `containers/Dockerfile.unified`
   - Docker tag: `felixmin/hlrp:unified-cuda-cu128`
   - Imported image naming: `hlrp_unified_cu128_imported_<timestamp>.sqsh`
3. Run `scripts/build_push_prune.sh --profile unified` from the workstation.
4. Submit the Enroot import with `scripts/submit_enroot_import.sh --profile unified`.
5. Monitor the returned Slurm job with `squeue` and `sacct`.
6. Before activation, check `squeue --me` and avoid moving the currently configured image while jobs are queued or running unless the user explicitly wants that.
7. Activate the new image with `scripts/swap_enroot_image.sh --replacement <new sqsh>`, which:
   - reads the currently configured image from `config/user_config/local.yaml`
   - moves the old `.sqsh` into `enroot/old/`
   - leaves the new `.sqsh` in `enroot/`
   - updates `config/user_config/local.yaml`
   - records provenance for the activation

## Rules

- Treat `unified` as the only first-class workflow for this skill.
- `stage12` / `stage3` helper behavior is legacy-only if it still exists in scripts for recovery.
- The imported unified image path stays dated and immutable; activation works by updating config, not by renaming the new image into a fixed target name.
- Default Enroot import settings should keep scratch/cache on DSS:
  - `TMPDIR`
  - `PARALLEL_TMPDIR`
  - `ENROOT_CACHE_PATH`
- Default Enroot import resource settings should remain `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`.
- `docker push` can sit in repeated `Waiting` state for a long time on large layers; do not treat that alone as a failure.
- Never delete cluster images or partial outputs. Rename or leave them in place.
- Activation must fail fast on nonstandard current-image states unless the caller explicitly uses the bootstrap override.

## Resources

- `scripts/build_push_prune.sh`
  Run the local Docker prune -> build -> push -> prune workflow for the unified profile.
- `scripts/submit_enroot_import.sh`
  Submit the high-memory `enroot import` job on `ssh ai` with DSS scratch/cache defaults and print the Slurm job id.
- `scripts/swap_enroot_image.sh`
  Archive the previously configured image into `enroot/old/` and update `config/user_config/local.yaml`.
- `references/workflow.md`
  Read this for the validated command shapes, default paths, and known failure modes.

## Validation

- Run the scripts with `--dry-run` first when adapting paths or tags.
- After editing the skill, run `quick_validate.py` on the skill folder.
