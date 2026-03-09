#!/usr/bin/env python3
"""
Script 1: Download Configured Datasets

Download the LeRobot datasets selected by the normal Hydra config composition.

Usage:
    python scripts/1_download_configured_datasets.py experiment=stage1_local data=test2
    python scripts/1_download_configured_datasets.py experiment=stage1_local data=test2 +download.limit=1
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add packages to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from common.cache_env import configure_cache_env, resolve_cache_dir


def _sort_sources(sources: list[dict]) -> list[dict]:
    indexed_sources = list(enumerate(sources))
    indexed_sources.sort(key=lambda item: (-float(item[1]["weight"]), item[0]))
    return [source for _, source in indexed_sources]


def _compose_hydra_cfg(*, overrides: list[str]) -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = str(workspace_root / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


def _repo_local_dir(root: Path, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return root / owner / name


def _is_downloaded(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists() and (path / "videos").exists()


def _resolve_explicit_root(cfg: DictConfig) -> Path | None:
    root_value = OmegaConf.select(cfg, "download.root")
    if root_value in (None, "", "null"):
        return None

    root = Path(str(root_value))
    if root.is_absolute():
        return root

    logging_root = Path(str(cfg.logging.root_dir)) if OmegaConf.select(cfg, "logging.root_dir") else workspace_root
    return logging_root / root


def _resolve_download_root(cfg: DictConfig) -> tuple[Path, str]:
    explicit_root = _resolve_explicit_root(cfg)
    if explicit_root is not None:
        return explicit_root, "download.root"

    import os

    resolved = os.environ.get("HF_LEROBOT_HOME")
    if not resolved:
        raise RuntimeError("HF_LEROBOT_HOME is not set after cache env setup")
    return Path(resolved), "HF_LEROBOT_HOME"


def main() -> None:
    cfg = _compose_hydra_cfg(overrides=sys.argv[1:])

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir)

    print("=" * 80)
    print("Dataset Download")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    if str(cfg.data.backend) != "lerobot_v3":
        raise ValueError(f"Only data.backend='lerobot_v3' is supported, got {cfg.data.backend!r}")

    root, root_source = _resolve_download_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Using download root: {root} ({root_source})")

    sources_cfg = OmegaConf.to_container(cfg.data.dataset.lerobot.sources, resolve=True)
    if not isinstance(sources_cfg, list):
        raise TypeError(type(sources_cfg))
    sources = _sort_sources(sources_cfg)

    start_from = OmegaConf.select(cfg, "download.start_from")
    if start_from not in (None, "", "null"):
        start_index = next(
            i for i, source in enumerate(sources) if str(source["repo_id"]) == str(start_from)
        )
        sources = sources[start_index:]

    limit = OmegaConf.select(cfg, "download.limit")
    if limit is not None:
        sources = sources[: int(limit)]

    dry_run = bool(OmegaConf.select(cfg, "download.dry_run"))

    print(f"Resolved {len(sources)} source(s) from Hydra config")
    for idx, source in enumerate(sources, start=1):
        repo_id = str(source["repo_id"])
        revision = source.get("revision")
        local_dir = _repo_local_dir(root, repo_id)
        print(
            f"[{idx}/{len(sources)}] repo_id={repo_id} revision={revision or 'default'} "
            f"weight={float(source['weight']):.2f} local_dir={local_dir}"
        )
        if _is_downloaded(local_dir):
            print("  already present, skipping")
            continue
        if dry_run:
            continue
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=["meta/*", "data/*", "videos/*", "README.md", ".gitattributes"],
            max_workers=4,
        )


if __name__ == "__main__":
    main()
