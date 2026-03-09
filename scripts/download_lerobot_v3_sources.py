from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


def _load_sources(config_path: str) -> list[dict]:
    cfg = OmegaConf.load(config_path)
    sources = OmegaConf.to_container(cfg["dataset"]["lerobot"]["sources"], resolve=True)
    if not isinstance(sources, list):
        raise TypeError(type(sources))
    indexed_sources = list(enumerate(sources))
    indexed_sources.sort(key=lambda item: (-float(item[1]["weight"]), item[0]))
    return [source for _, source in indexed_sources]


def _repo_local_dir(root: str, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return Path(root) / owner / name


def _is_downloaded(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists() and (path / "videos").exists()


def _resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("HF_LEROBOT_HOME")
    if not root:
        raise SystemExit(
            "Download root not provided. Pass `--root /path/to/huggingface/lerobot` "
            "or export `HF_LEROBOT_HOME` first."
        )
    return Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/octo24.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = _resolve_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Using download root: {root}")

    sources = _load_sources(args.config)
    if args.start_from is not None:
        start_index = next(i for i, source in enumerate(sources) if str(source["repo_id"]) == str(args.start_from))
        sources = sources[start_index:]
    if args.limit is not None:
        sources = sources[: int(args.limit)]

    for idx, source in enumerate(sources, start=1):
        repo_id = str(source["repo_id"])
        local_dir = _repo_local_dir(str(root), repo_id)
        print(f"[{idx}/{len(sources)}] repo_id={repo_id} weight={float(source['weight']):.2f} local_dir={local_dir}")
        if _is_downloaded(local_dir):
            print("  already present, skipping")
            continue
        if args.dry_run:
            continue
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["meta/*", "data/*", "videos/*", "README.md", ".gitattributes"],
            max_workers=4,
        )


if __name__ == "__main__":
    main()
