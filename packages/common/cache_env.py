from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    return Path(s)


def resolve_cache_dir(*, cfg: Any, workspace_root: Path) -> Path | None:
    """
    Resolve a shared cache directory for large artifacts.

    - Does NOT touch HF auth/token locations (we never set HF_HOME).
    - Intended for HF hub cache, torch hub cache, TFDS cache, etc.
    """
    cache_dir = _as_path(getattr(getattr(cfg, "paths", None), "cache_dir", None))
    if cache_dir is None:
        return None

    logging_root = _as_path(getattr(getattr(cfg, "logging", None), "root_dir", None))
    base_root = logging_root if logging_root is not None else workspace_root
    return cache_dir if cache_dir.is_absolute() else (base_root / cache_dir)


def configure_cache_env(*, cache_dir: Path, logger: Any | None = None) -> None:
    """
    Set env vars so large caches stay under `cache_dir`.

    Important:
    - We intentionally do NOT set `HF_HOME` to avoid breaking Hugging Face auth/token discovery.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_cache = cache_dir / "huggingface" / "hub"
    hf_datasets_cache = cache_dir / "huggingface" / "datasets"
    hf_lerobot_home = cache_dir / "huggingface" / "lerobot"
    torch_home = cache_dir / "torch"
    tfds_dir = cache_dir / "tfds"
    wandb_cache = cache_dir / "wandb_cache"

    for p in (
        hf_hub_cache,
        hf_datasets_cache,
        hf_lerobot_home,
        torch_home,
        tfds_dir,
        wandb_cache,
    ):
        p.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
    os.environ.setdefault("HF_LEROBOT_HOME", str(hf_lerobot_home))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("TFDS_DATA_DIR", str(tfds_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))

    if logger is not None:
        try:
            logger.info("Cache env configured (HF_HOME unchanged):")
            logger.info(f"  - HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
            logger.info(f"  - HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
            logger.info(f"  - HF_LEROBOT_HOME={os.environ.get('HF_LEROBOT_HOME')}")
            logger.info(f"  - TORCH_HOME={os.environ.get('TORCH_HOME')}")
            logger.info(f"  - TFDS_DATA_DIR={os.environ.get('TFDS_DATA_DIR')}")
            logger.info(f"  - WANDB_CACHE_DIR={os.environ.get('WANDB_CACHE_DIR')}")
        except Exception:
            pass


def hf_download_help_message(*, exc: BaseException) -> str | None:
    msg = str(exc)
    lowered = msg.lower()
    if (
        "401" in msg
        or "unauthorized" in lowered
        or "forbidden" in lowered
        or "gated" in lowered
    ):
        return (
            "Hugging Face download/auth may have failed (gated repo or missing token).\n"
            "- Run `huggingface-cli login` once on this machine.\n"
            "- Ensure you have accepted the model's terms if it is gated.\n"
            "- Do NOT override `HF_HOME` (auth lives there). We only redirect caches via `HF_HUB_CACHE`.\n"
            "- If needed, set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in your environment."
        )
    return None
