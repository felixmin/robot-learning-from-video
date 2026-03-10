from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.datasets.mixed_dataset import (
    _aggregate_selected_stats,
    _selected_episodes,
    build_explicit_mixed_stats,
    load_dataset_mix_config,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import serialize_dict

NORMALIZATION_STATS_FILENAME = "hlrp_normalization_stats.json"
TRAIN_CONFIG_FILENAME = "train_config.json"


def write_normalization_stats(
    *,
    save_directory: Path,
    stats: Mapping[str, Any],
) -> Path:
    save_directory.mkdir(parents=True, exist_ok=True)
    stats_path = save_directory / NORMALIZATION_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(serialize_dict(dict(stats)), f, indent=2, sort_keys=True)
    return stats_path


def load_saved_normalization_stats(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[Any, Any] | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    model_id = str(pretrained_name_or_path)
    stats_path: Path | None = None

    if Path(model_id).is_dir():
        candidate = Path(model_id) / NORMALIZATION_STATS_FILENAME
        if candidate.is_file():
            stats_path = candidate
    elif Path(model_id).is_file():
        candidate = Path(model_id)
        if candidate.name == NORMALIZATION_STATS_FILENAME and candidate.is_file():
            stats_path = candidate
    else:
        try:
            downloaded = hf_hub_download(
                repo_id=model_id,
                filename=NORMALIZATION_STATS_FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError:
            return None
        stats_path = Path(downloaded)

    if stats_path is None:
        return None

    with open(stats_path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected normalization stats mapping in {stats_path}, got {type(loaded).__name__}"
        )
    return loaded


def require_saved_normalization_stats(
    pretrained_name_or_path: str | Path,
    *,
    init_mode: str,
    **kwargs: Any,
) -> dict[str, dict[str, Any]] | None:
    saved_stats = load_saved_normalization_stats(pretrained_name_or_path, **kwargs)
    if saved_stats is None and str(init_mode) == "scratch":
        raise RuntimeError(
            "Scratch-mode HLRP checkpoint is missing "
            f"{NORMALIZATION_STATS_FILENAME}. Refusing to load a checkpoint with "
            "identity state/action normalization."
        )
    return saved_stats


def write_normalization_stats_from_train_config(policy_dir: Path) -> Path:
    train_config_path = policy_dir / TRAIN_CONFIG_FILENAME
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Missing train config: {train_config_path}")

    with open(train_config_path) as f:
        train_cfg = json.load(f)

    dataset_cfg = train_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(
            f"Expected 'dataset' mapping in {train_config_path}, got {type(dataset_cfg).__name__}"
        )

    mix_path = dataset_cfg.get("mix_path")
    if isinstance(mix_path, str) and mix_path:
        mix_file = Path(mix_path)
        if not mix_file.is_file():
            raise FileNotFoundError(f"Mixed dataset config not found: {mix_file}")
        mix_cfg = load_dataset_mix_config(mix_file)
        sources = []
        for source_cfg in mix_cfg.sources:
            ds_meta = LeRobotDatasetMetadata(
                repo_id=source_cfg.repo_id,
                root=source_cfg.root,
                revision=source_cfg.revision,
            )
            selected_episodes = _selected_episodes(source_cfg, ds_meta.total_episodes)
            selected_stats = _aggregate_selected_stats(ds_meta, selected_episodes)
            if selected_stats is None:
                raise RuntimeError(
                    f"Dataset stats are missing for mixed source repo_id={source_cfg.repo_id!r}"
                )
            sources.append(
                SimpleNamespace(
                    meta=SimpleNamespace(stats=selected_stats),
                    weight=float(source_cfg.weight),
                )
            )
        stats = build_explicit_mixed_stats(sources)
        return write_normalization_stats(save_directory=policy_dir, stats=stats)

    repo_id = dataset_cfg.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"Missing dataset.repo_id in {train_config_path}")

    ds_meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=dataset_cfg.get("root"),
        revision=dataset_cfg.get("revision"),
    )
    if ds_meta.stats is None:
        raise RuntimeError(f"Dataset stats are missing for repo_id={repo_id!r}")

    return write_normalization_stats(save_directory=policy_dir, stats=ds_meta.stats)
