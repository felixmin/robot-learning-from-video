# Paths and caching (local + cluster)

This repo writes *run artifacts* (logs/checkpoints/profiles/viz) into a run directory, and uses a separate *cache directory* for large downloaded artifacts (HF weights, torch hub, etc.).

## Run output directory

Configured via `logging.root_dir` / `logging.runs_dir`:

- `logging.runs_dir` (highest priority): write exactly there.
- else, `logging.root_dir`: write to `<logging.root_dir>/runs/<timestamp>_<experiment.name>/`.
- else: write to `<repo_root>/runs/<timestamp>_<experiment.name>/`.

W&B, checkpoints, visualizations, profiler outputs are all placed under the same run directory:

- `unified.log`
- `wandb/`
- `checkpoints/`
- `visualizations/`
- profiler outputs (e.g. `profiles/`)
- `.hydra/` (Hydra snapshots)

## Cache directory (large downloads)

Configured via `paths.cache_dir` (relative paths resolve against `logging.root_dir` if set, else repo root):

- HuggingFace hub weights: `HF_HUB_CACHE=<cache_dir>/huggingface/hub`
- HuggingFace datasets cache: `HF_DATASETS_CACHE=<cache_dir>/huggingface/datasets`
- LeRobot dataset / asset cache: `HF_LEROBOT_HOME=<cache_dir>/huggingface/lerobot`
- torch hub cache: `TORCH_HOME=<cache_dir>/torch`
- Triton cache: `TRITON_CACHE_DIR=<cache_dir>/triton`
- W&B cache: `WANDB_CACHE_DIR=<cache_dir>/wandb_cache`

For cluster stage-3 runs we keep:

- `HOME` at the real cluster home
- `HF_HOME=$HOME/.cache/huggingface`

and redirect the heavy caches above into DSS storage. This preserves token discovery while avoiding repeated large downloads into home directories.

## Hugging Face auth tokens

Hugging Face auth can come from:

- environment variables: `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
- tokens saved by `huggingface-cli login` in your home directory (default location)

If a model is gated and downloads fail (401/403), run `huggingface-cli login` and ensure you’ve accepted the model’s terms.

## Hydra output directory

Hydra creates a `.hydra/` snapshot directory per run. We configure Hydra to write its output into the same run directory as `unified.log` so you don’t get an extra `runs/output/hydra/...` tree.
