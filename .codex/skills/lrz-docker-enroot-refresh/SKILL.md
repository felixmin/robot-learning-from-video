---
name: lrz-docker-enroot-refresh
description: "Build and refresh the LRZ stage-1/2 or stage-3 container workflow end-to-end: infer the target profile from the task, build and push `felixmin/hlrp:stage12` or `felixmin/hlrp:stage3` from the matching Dockerfile, import the image to the LRZ cluster as an Enroot `.sqsh`, and safely swap it into `hlrp_stage12.sqsh` or `hlrp_stage3_lerobot.sqsh` by renaming backups in place."
---

# LRZ Docker Enroot Refresh

Use this workflow when refreshing the cluster container from the workstation. Infer the target from the user request and the stage context, keep local Docker steps on the workstation, keep cluster work behind `ssh ai`, never delete cluster images, and prefer renaming backups in place.

Read [workflow.md](references/workflow.md) for the defaults, paths, and failure modes. Use the scripts in `scripts/` rather than rewriting the commands.

## Workflow

1. Verify execution context with `pwd` and `hostname`.
2. Choose a profile:
   - `stage12` for LAQ, foundation, VLA, or stage-1/2 work
   - `stage3` for LeRobot, rollout, or stage-3 work
3. Run `scripts/build_push_prune.sh --profile <profile>` from the workstation.
4. Submit the Enroot import with `scripts/submit_enroot_import.sh --profile <profile>`.
5. Monitor the returned Slurm job with `squeue` and `sacct`.
6. If the import OOMs, keep the partial file, choose a new output path, and rerun with higher memory.
7. Before swapping images, check `squeue --me` and avoid replacing the active target while jobs are active unless the user explicitly wants that.
8. Swap the imported image into place with `scripts/swap_enroot_image.sh --profile <profile>`. This renames the existing stage-specific target to a timestamped backup and moves the new file into place.

## Rules

- Infer `stage12` for stage-1/2 requests and `stage3` for LeRobot/stage-3 requests. If the request is ambiguous, inspect the referenced experiment/config first and ask before swapping.
- `stage12` maps to `containers/Dockerfile.stage12`, `felixmin/hlrp:stage12`, and `.../enroot/hlrp_stage12.sqsh`.
- `stage3` maps to `containers/Dockerfile.stage3`, `felixmin/hlrp:stage3`, and `.../enroot/hlrp_stage3_lerobot.sqsh`.
- Treat OOM during `enroot import` as a memory tuning issue, not a registry issue.
- Default Enroot import settings should be `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`.
- Never delete cluster images or partial outputs. Rename or leave them in place.
- Do not let a stage-3 request update the stage-1/2 target, or vice versa.

## Resources

- `scripts/build_push_prune.sh`
  Run the local Docker prune -> build -> push -> prune workflow for either profile.
- `scripts/submit_enroot_import.sh`
  Submit the high-memory `enroot import` job on `ssh ai` for either profile and print the Slurm job id.
- `scripts/swap_enroot_image.sh`
  Rename the old stage-specific target to a timestamped backup and move the new image into place.
- `references/workflow.md`
  Read this for the validated command shapes, default paths, and known failure modes.

## Validation

- Run the scripts with `--dry-run` first when adapting paths or tags.
- After editing the skill, run `quick_validate.py` on the skill folder.
