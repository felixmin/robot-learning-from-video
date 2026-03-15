# Plan: Refresh LRZ Container Deployment Workflow For Unified Images

## 1. Change Summary

Refresh the LRZ container deployment workflow so the repo's operational docs, helper scripts, and skill instructions match the unified-container reality that is already documented in `containers/README.md` and used by cluster configs.

The target workflow is:

1. Build and push `containers/Dockerfile.unified` as the first-class image path.
2. Import the Docker Hub tag to a dated `.sqsh` under `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/`.
3. Keep all Enroot scratch/cache paths on DSS, not home.
4. On activation, move the previously referenced `.sqsh` into `enroot/old/`, leave the new `.sqsh` in `enroot/`, and update `config/user_config/local.yaml`.
5. Record provenance for each activation so historical image lookup remains deterministic.

## 2. Current Code And Docs Fit

### What already matches

- `containers/README.md` already says the unified container is the intended runtime for all stages.
- Shared cluster configs already expect `cluster.container.image` to come from user config rather than a hardcoded stage-specific path.
- The successful live import path is `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_unified_cu128_imported_2026-03-15_160458.sqsh`.

### What is stale or inconsistent

- `.codex/skills/lrz-docker-enroot-refresh/SKILL.md` still assumes only `stage12` and `stage3` profiles and a fixed-target swap step.
- `.codex/skills/lrz-docker-enroot-refresh/references/workflow.md` still documents `Dockerfile.stage12` / `Dockerfile.stage3`, fixed target paths, and stage-specific Enroot URIs.
- `.codex/skills/lrz-docker-enroot-refresh/scripts/build_push_prune.sh` only models `stage12` and `stage3` as first-class profiles.
- `.codex/skills/lrz-docker-enroot-refresh/scripts/submit_enroot_import.sh` originally defaulted scratch/cache paths to home and only became usable after live-turn fixes for:
  - `TMPDIR`
  - `PARALLEL_TMPDIR`
  - `ENROOT_CACHE_PATH`
- `.codex/skills/lrz-docker-enroot-refresh/scripts/swap_enroot_image.sh` encodes fixed-target stage-specific semantics that no longer match the unified workflow.
- `.codex/skills/lrz-docker-enroot-refresh/agents/openai.yaml` still tells the agent to infer stage12/stage3 and swap the image into place.
- `AGENTS.md` still says the cluster currently uses two separate stage-specific images and that unification is future work.
- `config/user_config/local.yaml.example` still documents only the imported-path override and not the new archive/activation semantics.

### Concrete blockers observed in the live run

1. Unified image build/push had to be forced through a `stage12` profile override because unified is not a real profile in the helper scripts.
2. First LRZ import failed because GNU `parallel` wrote buffers to `~/tmp`:
   - `Cannot append to buffer file in /dss/dsshome1/00/go98qik2/tmp`
3. Second LRZ import failed because Enroot still used a home cache path:
   - `mktemp: failed to create file via template '/dss/dsshome1/00/go98qik2/enroot/cache/...`
4. There is no scripted notion of "activate this dated unified import and archive the previously active one".
5. The workflow still talks about swapping fixed files like `hlrp_stage12.sqsh`, which is incompatible with `local.yaml`-driven unified deployment.

## 3. Design Decisions And Boundaries

### First-class workflow shape

- Treat `unified` as the only first-class deployment profile for this skill and workflow.
- Use:
  - Dockerfile: `containers/Dockerfile.unified`
  - Docker tag: `felixmin/hlrp:unified-cuda-cu128`
  - Imported image naming: `hlrp_unified_cu128_imported_<timestamp>.sqsh`
- Demote `stage12` / `stage3` to explicit legacy-only behavior.
  - They may remain temporarily in helper scripts only if needed for recovery.
  - They must be removed from the skill default path, operator docs, and agent prompt.

### Activation semantics

- The active cluster image for the normal operator workflow is whatever config file the activation helper is told to edit.
  - Default operator target: `config/user_config/local.yaml`
  - Test/dry-run target: explicit config path argument
- Activation is not a fixed-path rename anymore, and it must be config-driven rather than hidden script state.
- Activation must:
  1. accept an explicit config path argument, defaulting to `config/user_config/local.yaml`
  2. read the current configured image path from that config file
  2. verify the new imported `.sqsh` exists
  3. fail fast if the current configured image path is outside `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/`, unless an explicit bootstrap override is supplied
  4. treat bootstrap states explicitly:
     - missing config file: fail unless bootstrap override is supplied
     - missing `cluster.container.image` / null image: allow only with bootstrap override, then update to the new image without archive
     - current image already equals new image: no-op and report that activation is already current
     - current image path missing on disk: fail by default, or allow bootstrap override to continue without archive
  5. create `.../enroot/old/` if needed
  6. move the previously referenced `.sqsh` into `old/` using deterministic naming:
     - preserve original stem
     - append activation timestamp suffix
     - keep `.sqsh`
  7. write a provenance record that maps:
     - previous active path
     - archived path
     - new active path
     - docker tag
     - pushed digest
     - import job id
  8. update the target config file to the new image path

### Scratch/cache ownership

- All Enroot import scratch/cache locations must live on DSS by default.
- `submit_enroot_import.sh` should own these defaults directly; the user should not need to remember ad hoc env vars.
- Keep defaults fail-fast and explicit:
  - `TMPDIR`
  - `PARALLEL_TMPDIR`
  - `ENROOT_CACHE_PATH`

### Backward-compatibility boundary

- Do not keep misleading split-image behavior as the default path.
- `stage12` / `stage3` are no longer canonical for this workflow.
- Keep legacy split-image behavior only as an explicit recovery mode if implementation needs a short transition period.
- Remove or replace old fixed-target swap semantics from normal entrypoints instead of preserving them silently.

## 4. Concrete Implementation Steps

### Step 1: Make the unified profile first-class in helper scripts

- Update `.codex/skills/lrz-docker-enroot-refresh/scripts/build_push_prune.sh`
  - add `unified` profile
  - default image to `felixmin/hlrp:unified-cuda-cu128`
  - default Dockerfile to `containers/Dockerfile.unified`
  - update help text and profile validation
- Update `.codex/skills/lrz-docker-enroot-refresh/scripts/submit_enroot_import.sh`
  - add `unified` profile
  - default Enroot URI/output/job name for unified imports
  - keep DSS-based `TMPDIR`, `PARALLEL_TMPDIR`, and `ENROOT_CACHE_PATH` as mandatory defaults
  - make the DSS scratch/cache path selection explicit in help output

### Step 2: Replace stale activation semantics

- Replace `.codex/skills/lrz-docker-enroot-refresh/scripts/swap_enroot_image.sh` in place with a config-driven activation/archive helper.
  - Keep the path if convenient for continuity, but change its behavior and help text to match unified deployment.
  - Do not keep the old fixed-target swap semantics behind the same default interface.
  - Required behavior:
    - validate new imported `.sqsh`
    - accept `--config-path` with default `config/user_config/local.yaml`
    - read previous `cluster.container.image` from that config
    - move previous image into `enroot/old/` with deterministic timestamped archive naming
    - write provenance metadata/note
    - update the target config file
    - guard on queued or running jobs unless explicitly overridden
    - provide explicit bootstrap override for missing/nonstandard prior state
- Keep moves only; never delete archived images.

### Step 3: Update the skill and workflow docs

- Update `.codex/skills/lrz-docker-enroot-refresh/SKILL.md`
  - unify the workflow around build -> push -> import -> archive old -> update config
  - mention DSS scratch/cache defaults
  - remove fixed-target stage-specific swap language from the primary instructions
  - mark split-image behavior as legacy-only if it remains anywhere
- Update `.codex/skills/lrz-docker-enroot-refresh/agents/openai.yaml`
  - align the skill summary and default prompt with the unified import/archive/update flow
  - remove stage12/stage3 inference as the default behavior
- Update `.codex/skills/lrz-docker-enroot-refresh/references/workflow.md`
  - replace old command shapes with unified image examples
  - document the actual live failure signatures we hit and the validated fix
  - document `enroot/old/` archive behavior

### Step 4: Update repo-facing docs and operator notes

- Update `containers/README.md`
  - add the archive-and-activate step after import
  - mention that imports must keep scratch/cache on DSS
  - mention where provenance for an activation is recorded
- Update `AGENTS.md`
  - remove the outdated "two separate containers are used" statement
  - rewrite container/image notes around the unified image
  - describe the active-image/user-config/archive model
- Update `config/user_config/local.yaml.example`
  - keep the imported-path example
  - document that `local.yaml` is the default activation target but helpers can operate on an explicit config path
- Review other operator docs that still advertise fixed stage-specific paths and update only the actively used ones.
  - likely candidate: `docs/felix_notes/general/job_submission.md`
  - leave clearly historical/recovery notes alone unless they are presented as current workflow

### Step 5: Validate the new workflow end to end

- Dry-run each helper script in unified mode.
- Run one real import using the unified helpers.
- Run one real activation using the new helper against a testable config path first, then against the real operator config.
- Verify:
  - the old image moves into `enroot/old/`
  - the new image stays in `enroot/`
  - the target config updates to the new path
  - provenance is recorded with old/new/archive/tag/digest/job metadata

## 5. Test And Validation Strategy

### Script-level validation

- `build_push_prune.sh --dry-run --profile unified`
- `submit_enroot_import.sh --dry-run --profile unified`
- activation helper `--dry-run` and real run against an explicit temp config file, covering:
  - normal case with an existing old image path
  - already-current case
  - bootstrap case with null/missing image
  - outside-DSS-enroot current path
  - missing current image on disk
- verify `config/user_config/local.yaml.example` matches the new operator model

### Real operational validation

- One real LRZ import of the unified image using DSS scratch/cache defaults.
- One real activation of the newly imported image.
- Confirm final Slurm status with `sacct`.
- Confirm the imported `.sqsh` exists and is non-trivial in size.
- Confirm archived prior image lands in `enroot/old/`.
- Confirm the real operator config points to the new imported image.
- Confirm `scripts/submit_job.py` resolves the activated image path through the normal config path, either via dry-run submission output or equivalent resolution path.

### Regression checks

- Verify no current docs or skill text still instruct users to swap `hlrp_stage12.sqsh` / `hlrp_stage3_lerobot.sqsh` as the default unified flow.
- Verify the skill scripts no longer require the stage12 override hack to build/push/import the unified image.
- Verify queued jobs are treated as blockers for activation, not only running jobs.

## 6. Documentation And Progress-Note Impact

- Canonical operator-facing updates:
  - `AGENTS.md`
  - `containers/README.md`
  - `config/user_config/local.yaml.example`
  - `.codex/skills/lrz-docker-enroot-refresh/SKILL.md`
  - `.codex/skills/lrz-docker-enroot-refresh/agents/openai.yaml`
  - `.codex/skills/lrz-docker-enroot-refresh/references/workflow.md`
- Add one dated note to `docs/felix_notes/` or the relevant progress note location summarizing:
  - home tmp failure
  - home Enroot cache failure
  - DSS scratch/cache fix
  - archive-and-activate workflow change
  - docker tag and pushed digest
  - import job id
  - previous active path -> archived path -> new active path mapping

## 7. Risks, Cleanup, And Open Questions

### Risks

- Archiving the previously configured image while jobs are queued or running could break workflows that still expect that exact path to remain in place.
  - activation helper should block on queued or running jobs by default
- `enroot/old/` will grow over time because deletion is forbidden by cluster policy.
  - this is acceptable, but the workflow should make the accumulation visible
- If a target config references an image outside the expected DSS Enroot root, the archive step becomes ambiguous.
  - default behavior should be fail-fast unless bootstrap override is explicit

### Open questions that matter

1. Provenance storage format: append to a single dated operator note, or write a small machine-readable provenance file under `enroot/old/`.
   - recommended default: write a machine-readable sidecar in `enroot/old/` and summarize in the dated operator note
2. Whether to keep a temporary legacy CLI surface for split-image recovery after the unified path lands.
   - recommended default: if retained, mark as legacy-only in help text and remove from skill/docs

## 8. Legacy-Code Removal And Hydra-Default Ownership

- Remove or demote code and docs that imply fixed stage-specific target images are still the default deployment path.
- Do not add Python fallbacks for missing image paths; `cluster.container.image` remains owned by config and should fail explicitly when absent.
- Keep activation state in the chosen config file, with `config/user_config/local.yaml` as the operator default, not in hidden script-local state.
- Prefer deleting obsolete swap-only logic over maintaining parallel workflows that disagree about what "deployment" means.
- Keep Hydra override reality explicit in docs:
  - `local.yaml` is the default operator path
  - ad hoc CLI overrides can still bypass it for one-off submissions
  - the deployment helper manages the default operator config, not every possible invocation

## Ordered Rollout

1. Update helper scripts to support `unified` and DSS scratch/cache paths.
2. Replace `swap_enroot_image.sh` in place with config-driven archive-and-update semantics for unified deployment.
3. Update skill docs, skill agent prompt, and workflow references to the unified path.
4. Update repo-level operator docs (`AGENTS.md`, `containers/README.md`, `config/user_config/local.yaml.example`, and any still-active container notes).
5. Run dry-runs, then one real import + one real activation validation, including `submit_job.py` image resolution.
6. Record the workflow change and activation provenance in a dated progress note/operator note.
