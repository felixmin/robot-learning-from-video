# Cluster container notes

This note records the current cluster container setup, the stage-3 debugging steps taken so far, the exact failure signatures we saw, and the issues that were fixed on the way.

## Current setup

### Stage split

- Stage 1/2 image: `hlrp_stage12.sqsh`
- Stage 3 image: `hlrp_stage3_lerobot.sqsh`

Stage-3 runs use the dedicated LeRobot image and the container venv:

- `cluster.container.python_bin=/opt/lerobot-venv/bin/python`

### Stage-3 cache and home behavior

Stage-3 submission currently uses:

- `HOME` -> real cluster home
- `HF_HOME=$HOME/.cache/huggingface`
- `HF_HUB_CACHE` -> DSS cache
- `HF_DATASETS_CACHE` -> DSS cache
- `HF_LEROBOT_HOME` -> DSS cache
- `TORCH_HOME` -> DSS cache
- `TRITON_CACHE_DIR` -> DSS cache

This keeps auth and user config in the real home directory while pushing large caches and dataset downloads to DSS.

### Stage-3 runtime assumptions

The current stage-3 flow assumes:

- the stage-3 image already contains a working LeRobot runtime
- the repo can be mounted into the container
- stage-3 jobs can optionally editable-install mounted packages
- HLRP stage-3 jobs currently editable-install:
  - `lerobot`
  - `lerobot_policy_hlrp`

## What was fixed

These problems were encountered and fixed during the stage-3 container work.

### Wrong container targeting

Problem:

- the earlier refresh workflow replaced `lam.sqsh` even when the intended target was the stage-3 image

Fix:

- stage-1/2 and stage-3 images were split explicitly
- refresh workflow now targets:
  - `hlrp_stage12.sqsh`
  - `hlrp_stage3_lerobot.sqsh`

### Wrong Python at runtime

Problem:

- stage-3 jobs were not reliably using the Python environment that actually contained the LeRobot dependencies

Symptoms:

- `ModuleNotFoundError: hydra`
- `ModuleNotFoundError: lerobot.scripts`

Fix:

- stage-3 submission now uses `/opt/lerobot-venv/bin/python`

### Broken stage-3 venv population

Problem:

- an earlier stage-3 image rebuild produced a venv that existed but was missing required packages

Fix:

- stage-3 image install flow was changed to a working venv setup
- the image now passes build-time imports for:
  - `hydra`
  - `omegaconf`
  - `lerobot`
  - `lerobot.scripts`

### Editable install path failures

Problem:

- stage-3 overlay runs failed because the cluster-side mounted `lerobot/` checkout was not actually present in usable form

Fix:

- the actual `lerobot/` source tree was synced to cluster
- stage-3 jobs now successfully editable-install:
  - `lerobot`
  - `lerobot_policy_hlrp`

### Early EGL permission failure

Problem:

- earlier stage-3 runs with `MUJOCO_GL=egl` failed during Libero env creation

Symptoms:

- `failed to open /dev/dri/renderD131: Permission denied`
- `failed to open /dev/dri/card4: Permission denied`
- `ImportError: Cannot initialize a EGL device display`

Fix:

- stage-3 Libero configs were switched from `egl` to `osmesa`

### Interactive LIBERO prompt

Problem:

- stock LeRobot runs could reach `Creating env` and then fail because LIBERO tried to ask for a dataset/config path interactively

Symptom:

- `EOFError: EOF when reading a line`

Fix:

- non-interactive LIBERO bootstrap was added by precreating a `LIBERO_CONFIG_PATH` config file before `lerobot-train`
- this removed the interactive prompt failure

### Wrong cache location and disk quota pressure

Problem:

- stage-3 runs were writing large caches into the container user home, which led to quota problems and unnecessary repeated downloads

Fix:

- cache ownership was moved to the submit layer
- heavy caches now go to DSS
- `HOME` remains the real cluster home so HF auth still works

## What the current failure is

The current unresolved failure is a native crash during Libero / robosuite env creation under `osmesa`.

Important properties:

- it happens after dataset creation
- it happens at `Creating env`
- it appears in both:
  - HLRP stage-3 runs
  - stock LeRobot runs
- it therefore does not currently look specific to:
  - the HLRP policy
  - editable-installing local `lerobot`
  - the stage-3 wrapper script itself

The strongest current hypothesis is:

- the remaining problem is in the stage-3 image's native graphics / LeRobot / robosuite / Libero runtime stack, not in HLRP policy code

## New image-mismatch finding

After the source-install stage-3 rebuild, the current Docker Hub tag and the live cluster image diverged.

What was observed:

- local `docker run felixmin/hlrp:stage3 ...` shows:
  - `lerobot 0.4.4`
  - `PyOpenGL 3.1.10`
  - `PyOpenGL-accelerate 3.1.10`
  - `hf-libero 0.1.3`
  - `egl_probe 1.0.2`
- the live cluster image `hlrp_stage3_lerobot.sqsh` shows:
  - `lerobot` import works
  - `hydra` import works
  - `omegaconf` import works
  - `robosuite` import works
  - `libero` import works
  - `PyOpenGL-accelerate` is missing

The missing package was verified directly inside the live cluster image:

- `python -c "import OpenGL_accelerate"` fails
- `pip show PyOpenGL-accelerate` reports package not found

But the same live cluster image can install it immediately:

- `pip install PyOpenGL-accelerate==3.1.10` succeeds from a prebuilt wheel

Interpretation:

- the cluster runtime is not missing compiler or system dependencies for this package
- the live `.sqsh` most likely does not match the final Docker Hub tag contents
- the most likely explanation is that the stage-3 image was imported from the registry before the push had fully settled

Current action:

- a fresh stage-3 Enroot import from the current Docker Hub tag was submitted
- before swapping it live, the imported `.sqsh` should be validated with the same tiny package check

## Run comparison

### Run `5499685`

Type:

- HLRP stage-3 probe
- older stage-3 image path
- `osmesa`

Status:

- `FAILED`
- exit code `1:0`
- elapsed `00:10:45`

What happened:

1. dataset creation started
2. large Libero fetch ran
3. `Creating env`
4. robosuite macro warnings
5. `No OpenGL_accelerate module loaded`
6. process died with `SIGSEGV`

Key point:

- this is the earliest clear `osmesa` segfault signature we saw on the HLRP path

### Run `5499945`

Type:

- stock LeRobot `ACT`
- rebuilt stage-3 image
- direct `lerobot-train`
- before non-interactive LIBERO bootstrap was added

Status:

- `FAILED`
- exit code `1:0`
- elapsed `00:09:44`

What happened:

1. dataset creation started
2. full Libero download completed
3. `Creating env`
4. import entered `libero.libero`
5. LIBERO tried to ask:
   - `Do you want to specify a custom path for the dataset folder?`
6. job failed with:
   - `EOFError: EOF when reading a line`

Key point:

- this was not a segfault
- this was an interactive bootstrap problem

### Run `5499946`

Type:

- HLRP stage-3 probe
- rebuilt stage-3 image
- editable install of local `lerobot` and `lerobot_policy_hlrp`

Status:

- `FAILED`
- exit code `1:0`
- elapsed `00:12:49`

What happened:

1. container startup succeeded
2. editable install of `lerobot` succeeded
3. editable install of `lerobot_policy_hlrp` succeeded
4. `lerobot-train` launched
5. full Libero dataset download completed
6. `Creating env`
7. robosuite macro warnings
8. `No OpenGL_accelerate module loaded`
9. process died with `SIGSEGV`

Key point:

- the rebuilt image fixed earlier startup and import problems
- but did not fix the native env-creation crash on the HLRP path

### Run `5500304`

Type:

- stock LeRobot `ACT`
- rebuilt stage-3 image
- non-interactive LIBERO bootstrap
- `osmesa`

Status:

- `FAILED`
- exit code `11:0`
- elapsed `00:00:50`

What happened:

1. non-interactive LIBERO config file was written successfully
2. dataset creation started
3. `Creating env`
4. robosuite macro warnings
5. `No OpenGL_accelerate module loaded`
6. process died with:
   - `Segmentation fault (core dumped)`

Key point:

- this removed the old `EOFError`
- the remaining failure now matches the earlier HLRP `osmesa` segfault pattern

### Run `5500387`

Type:

- stock LeRobot `ACT`
- source-install stage-3 rebuild
- non-interactive LIBERO bootstrap
- `osmesa`

Status:

- `FAILED`
- exit code `11:0`
- elapsed `00:00:43`

What happened:

1. non-interactive LIBERO config file was written successfully
2. dataset creation started
3. `Creating env`
4. robosuite macro warnings
5. `No OpenGL_accelerate module loaded`
6. process died with:
   - `Segmentation fault (core dumped)`

Key point:

- moving stage 3 to a source-install LeRobot image did not remove the segfault in the live cluster image
- however, later image diagnostics showed that this live `.sqsh` was still missing `PyOpenGL-accelerate`, so the run may still have been using a stale import of the tag

## Similarities and differences between the key runs

### What is the same in the segfaulting runs

Common signature across `5499685`, `5499946`, `5500304`, and `5500387`:

1. dataset setup succeeds
2. run reaches `Creating env`
3. robosuite prints the same macro warnings
4. log contains:
   - `No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'`
5. process dies with `SIGSEGV` / exit `11`

Interpretation:

- the crash happens in or immediately after Libero / robosuite env creation
- it is not specific to HLRP policy code
- it is not specific to editable overlays

### What is different

`5499945` differs from the others:

- it did not reach the segfault
- it failed earlier with interactive LIBERO bootstrap

`5499946` differs from `5500304`:

- `5499946` uses the HLRP policy path with editable overlays
- `5500304` is stock `ACT` with no HLRP policy/plugin involvement

Interpretation:

- since both paths hit the same env-creation segfault after bootstrap is fixed, the custom policy is not the leading suspect

`5499685` differs from `5499946`, `5500304`, and `5500387`:

- it happened on an older stage-3 image revision

Interpretation:

- the rebuild fixed several surrounding issues, but the native env-creation crash remained

`5500387` adds one more important nuance:

- it used the source-install stage-3 rebuild
- but the live cluster image later turned out to be missing `PyOpenGL-accelerate`, while the current Docker Hub tag contains it

Interpretation:

- `5500387` may not actually represent the final rebuilt tag state
- the next clean probe should use a freshly re-imported `.sqsh` that has been package-checked first

## Other issues encountered during analysis

These occurred during the debugging process but are not the main current blocker.

### LRZ container-start failures

Some LRZ attempts failed before Python started.

Symptoms:

- `pyxis: container start failed`
- `nvidia-container-cli: detection error: nvml error: unknown error`

Interpretation:

- node/runtime instability
- not directly informative for the Libero crash

### Output directory collisions

Some stage-3 smoke runs failed because LeRobot would not reuse an existing output directory when `resume=false`.

Fix:

- use a subdirectory under the run dir, not the run dir root

### Slow first cold-start

The first run after moving cache to DSS is slow because Libero pulls a large payload into `HF_LEROBOT_HOME`.

This is expected.

## Current conclusion

What is now clearly fixed:

- image targeting
- stage-3 Python selection
- stage-3 import/runtime startup
- editable install path
- cache placement
- early EGL permission failure
- interactive LIBERO prompt

What is still not fixed:

- native Libero / robosuite env creation under the current stage-3 image

What is also not yet settled:

- whether the latest source-install stage-3 Docker tag itself still segfaults once the cluster `.sqsh` is re-imported correctly

Current best read:

- the remaining blocker is below the HLRP policy layer
- the remaining blocker is also below the editable local `lerobot` overlay
- the next meaningful step is a more fundamental stage-3 image change, likely aligning more closely with the source-install LeRobot Dockerfile that is known to work better

## Suggested next steps

1. Finish the fresh Enroot import from the current `felixmin/hlrp:stage3` tag.
2. Validate the imported `.sqsh` directly:
   - `import OpenGL_accelerate`
   - `pip show PyOpenGL-accelerate`
3. Swap that validated image live.
4. Rerun the stock non-interactive `ACT` Libero probe.
5. Only if stock still segfaults on the validated image, continue deeper stage-3 graphics/runtime debugging.
