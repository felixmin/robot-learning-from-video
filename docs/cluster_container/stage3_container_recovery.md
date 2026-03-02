# Stage 3 Container Recovery

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

## What the final root causes were

The stage-3 failures turned out to be a stack of separate issues rather than one bug.

### Root cause 1: stale / incomplete stage-3 image content

At one point the imported cluster `.sqsh` did not match the intended Docker tag contents.

What this looked like:

- source-install stage-3 rebuild still segfaulted at `Creating env`
- `OpenGL_accelerate` was missing in the live cluster image
- the same package existed in the Docker tag and installed cleanly when added manually

Root cause:

- the live cluster image had been imported from a stale/incomplete registry state

Fix:

- rebuild stage 3 again
- re-import from the settled `felixmin/hlrp:stage3` tag
- validate the imported `.sqsh` directly before swapping live

### Root cause 2: broken LeRobot package discovery in the nested `lerobot` repo

What this looked like:

- `ModuleNotFoundError: No module named 'lerobot.scripts'`
- image had `lerobot-0.4.4.dist-info` but not the actual package files

Root cause:

- `lerobot/pyproject.toml` package discovery for the `src/` layout was incomplete
- `lerobot/scripts` also needed to be a real package

Fix:

- add `package-dir = {\"\" = \"src\"}` and `include = [\"lerobot*\"]` in [lerobot/pyproject.toml](/mnt/data/workspace/code/high-level-robot-planner/lerobot/pyproject.toml)
- add [lerobot/src/lerobot/scripts/__init__.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/scripts/__init__.py)

### Root cause 3: mounted repo shadowing the installed package

What this looked like:

- the image could validate in one context and fail in another
- `lerobot` import behavior depended on the working directory

Root cause:

- stage-3 commands were launched from the mounted repo root, which shadowed the installed package layout

Fix:

- launch stage-3 train and rollout from `logging.runs_dir`, not from the repo root
- implemented in:
  - [scripts/6_train_lerobot.py](/mnt/data/workspace/code/high-level-robot-planner/scripts/6_train_lerobot.py)
  - [scripts/7_rollout_lerobot.py](/mnt/data/workspace/code/high-level-robot-planner/scripts/7_rollout_lerobot.py)

### Root cause 4: Libero bootstrap and graphics config issues around the main failure

These were separate from the final container issue but blocked diagnosis.

Problems:

- `MUJOCO_GL=egl` caused `/dev/dri/*` permission failures
- stock LeRobot runs prompted interactively for LIBERO config paths

Fixes:

- switch stage-3 Libero runs to `MUJOCO_GL=osmesa`
- precreate `LIBERO_CONFIG_PATH` for non-interactive jobs

### Root cause 5: HLRP launcher bug after the container/runtime fix

What this looked like:

- HLRP run failed before `lerobot-train` launched
- traceback in [scripts/6_train_lerobot.py](/mnt/data/workspace/code/high-level-robot-planner/scripts/6_train_lerobot.py) on `int(None)` for `grad_accum_steps`

Root cause:

- the launcher emitted `--grad_accum_steps=...` even when the config value was `null`

Fix:

- keep `grad_accum_steps` support, but only append the CLI flag when the config actually sets a value

## What finally proved the fix

### Stock LeRobot proof run

Run:

- `5500539`

What it proved:

- stage-3 image can create all Libero envs under `osmesa`
- stock LeRobot can create policy, optimizer, and start training
- the old `Creating env` segfault is gone

### HLRP eval-enabled proof run

Run:

- `5500555`

What it proved:

- editable install of `lerobot` works
- editable install of `lerobot_policy_hlrp` works
- HLRP Libero env creation works
- HLRP policy creation works
- HLRP training reaches step 10
- the remaining failure there was only the 15-minute smoke budget expiring during eval

### HLRP train-only proof run

Run:

- `5500588`

Status:

- `COMPLETED`

What it proved:

- full HLRP stage-3 training path works end to end on cluster
- run reached:
  - training start
  - step 10
  - step 20
  - checkpoint save
  - end of training

## Current state

Resolved:

- stage-3 image targeting
- stage-3 Python selection
- stage-3 cache/home behavior
- non-interactive LIBERO bootstrap
- `egl` device-permission failure
- stale/incomplete stage-3 image imports
- missing `OpenGL_accelerate`
- LeRobot package discovery / `lerobot.scripts` packaging
- mounted-repo shadowing of installed LeRobot
- HLRP editable overlay startup
- HLRP training startup and training-only smoke execution

Not yet fully verified in this note:

- the 30-minute eval-enabled HLRP smoke that was submitted after the training-only confirmation

Current submitted eval-enabled runs:

- LRZ: `5500594`
- MCML: `5500595`

## Historical note

The earlier segfaulting runs in this document are still useful as failure history, but they no longer describe the current state of the stage-3 system. The main stage-3 container/runtime issue is resolved; the remaining verification task is only to confirm the longer eval-enabled smoke budget.

## Recovery checklist

If the stage-3 path breaks again, use this order.

1. Validate the symptom first.
   - `EOFError: EOF when reading a line` at LIBERO startup:
     missing non-interactive `LIBERO_CONFIG_PATH` bootstrap
   - `/dev/dri/* permission denied` or `Cannot initialize a EGL device display`:
     wrong `MUJOCO_GL=egl` path, switch back to `osmesa`
   - `ModuleNotFoundError: lerobot.scripts`:
     broken `lerobot` packaging or runtime cwd shadowing
   - `SIGSEGV` at `Creating env` with `OpenGL_accelerate` missing:
     stale or incomplete stage-3 image import

2. Validate the Docker tag locally before importing.
   - confirm the stage-3 image builds successfully
   - confirm the image-side Python check passes for:
     - `lerobot`
     - `lerobot.scripts.lerobot_train`
     - `hydra`
     - `omegaconf`
     - `OpenGL_accelerate`

3. Import a fresh stage-3 `.sqsh` from the current Docker tag.
   - do not swap it live yet

4. Validate the imported `.sqsh` from a neutral cwd such as `/tmp`.
   - required imports:
     - `lerobot`
     - `lerobot.scripts.lerobot_train`
     - `hydra`
     - `omegaconf`
     - `OpenGL_accelerate`

5. Only after that, swap the validated `.sqsh` into `hlrp_stage3_lerobot.sqsh`.

6. Run the stock discriminator first.
   - stock `ACT`
   - non-interactive LIBERO bootstrap
   - `MUJOCO_GL=osmesa`
   - no HLRP policy/plugin

7. If stock works, run the HLRP smoke next.
   - first with `lerobot.eval.freq=0` for a pure training smoke
   - then with eval enabled and a longer time limit if needed

8. If HLRP fails before `lerobot-train` launches, check the wrapper layer.
   - command construction in [scripts/6_train_lerobot.py](/mnt/data/workspace/code/high-level-robot-planner/scripts/6_train_lerobot.py)
   - especially optional flags such as `grad_accum_steps`

9. If `lerobot` imports change depending on cwd, check the mounted-repo shadowing rule.
   - stage-3 train and rollout must run from `logging.runs_dir`, not the repo root
