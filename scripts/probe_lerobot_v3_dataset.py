#!/usr/bin/env python3
"""
Probe a LeRobot v3 dataset against the HLRP Stage-1/Stage-2 data contract.

This script is intentionally conservative:
- metadata checks are read-only apart from small cached downloads,
- runtime probing is opt-in because video-backed datasets may download chunk files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


DEFAULT_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "data/chunk-000/file-000.parquet",
)


def _sha256_prefix(path: Path, length: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:length]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def summarize_repo_metadata(repo_id: str, cache_root: Path) -> dict[str, Any]:
    repo_root = cache_root / repo_id.replace("/", "__")
    repo_root.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for rel_path in DEFAULT_FILES:
        downloaded[rel_path] = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rel_path,
                local_dir=repo_root,
            )
        )

    info = json.loads(downloaded["meta/info.json"].read_text())
    tasks = pd.read_parquet(downloaded["meta/tasks.parquet"])
    parquet = pq.read_table(downloaded["data/chunk-000/file-000.parquet"])

    features = info.get("features", {})
    camera_keys = [
        k for k, v in features.items() if v.get("dtype") in {"image", "video"}
    ]

    return {
        "repo_id": repo_id,
        "fingerprints": {
            rel_path: {
                "sha256_prefix": _sha256_prefix(path),
                "size_bytes": path.stat().st_size,
            }
            for rel_path, path in downloaded.items()
        },
        "info": {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "camera_keys": camera_keys,
            "state_shape": features.get("observation.state", {}).get("shape"),
            "state_names": features.get("observation.state", {}).get("names"),
            "action_shape": features.get("action", {}).get("shape"),
            "action_names": features.get("action", {}).get("names"),
        },
        "tasks": {
            "count": int(len(tasks)),
            "examples": [str(x) for x in list(tasks.index[:5])],
        },
        "first_parquet_shard": {
            "columns": parquet.column_names,
            "rows": parquet.num_rows,
            "first_row_meta": {
                k: v
                for k, v in parquet.slice(0, 1).to_pylist()[0].items()
                if k
                in {"timestamp", "episode_index", "frame_index", "task_index", "index"}
            },
        },
    }


def compare_repo_metadata(
    primary: dict[str, Any], other: dict[str, Any]
) -> dict[str, Any]:
    rel_paths = sorted(set(primary["fingerprints"]) | set(other["fingerprints"]))
    file_matches: dict[str, bool] = {}
    for rel_path in rel_paths:
        p = primary["fingerprints"].get(rel_path)
        q = other["fingerprints"].get(rel_path)
        file_matches[rel_path] = p == q

    return {
        "primary_repo_id": primary["repo_id"],
        "compare_repo_id": other["repo_id"],
        "all_files_match": all(file_matches.values()),
        "file_matches": file_matches,
        "primary_info": primary["info"],
        "compare_info": other["info"],
    }


def summarize_local_oxe_first_episode(
    dataset_name: str, local_root: Path
) -> dict[str, Any]:
    dataset_root = local_root / dataset_name
    tar_paths = sorted(dataset_root.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No tar shards found under {dataset_root}")

    tar_path = tar_paths[0]
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".data.pickle"):
                continue
            fileobj = tf.extractfile(member)
            if fileobj is None:
                raise RuntimeError(f"Failed to extract {member.name} from {tar_path}")
            episode = pickle.load(fileobj)
            step = episode["steps"][0]
            obs = step["observation"]

            instruction = obs.get("natural_language_instruction") or obs.get(
                "language_instruction"
            )
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8", errors="replace")

            action = step.get("action")
            action_summary: dict[str, Any]
            if isinstance(action, dict):
                action_summary = {"keys": sorted(str(k) for k in action.keys())}
            else:
                action_summary = {
                    "type": type(action).__name__,
                    "shape": tuple(action.shape) if hasattr(action, "shape") else None,
                }

            tensor_shapes = {
                str(k): tuple(v.shape)
                for k, v in obs.items()
                if hasattr(v, "shape") and str(k) not in {"natural_language_embedding"}
            }
            byte_keys = [
                str(k)
                for k, v in obs.items()
                if isinstance(v, (bytes, bytearray))
                and str(k) != "natural_language_instruction"
            ]

            return {
                "dataset_name": dataset_name,
                "first_tar": str(tar_path),
                "num_tar_files": len(tar_paths),
                "first_episode_steps": int(len(episode["steps"])),
                "instruction": instruction,
                "observation_keys": sorted(str(k) for k in obs.keys()),
                "byte_like_keys": byte_keys,
                "tensor_shapes": tensor_shapes,
                "action_summary": action_summary,
            }

    raise RuntimeError(f"No episode payload found in {tar_path}")


def probe_runtime(
    *,
    repo_id: str,
    camera_key: str,
    episode: int,
    pair_offset_steps: int,
    action_chunk_size: int,
    download_root: Path,
    video_backend: str,
) -> dict[str, Any]:
    root = download_root / repo_id.replace("/", "__")

    meta_only = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[int(episode)],
        download_videos=False,
        video_backend=video_backend,
    )
    fps = int(meta_only.fps)
    del meta_only

    delta_timestamps = {
        camera_key: [0.0, float(pair_offset_steps) / float(fps)],
        "observation.state": [0.0],
        "action": [float(i) / float(fps) for i in range(int(action_chunk_size))],
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[int(episode)],
        delta_timestamps=delta_timestamps,
        download_videos=True,
        video_backend=video_backend,
    )

    first = dataset[0]
    last = dataset[len(dataset) - 1]
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    def item_summary(item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in item.items():
            if hasattr(value, "shape"):
                out[key] = {
                    "shape": tuple(int(x) for x in value.shape),
                    "dtype": str(getattr(value, "dtype", "")),
                }
            elif isinstance(value, str):
                out[key] = value
            else:
                out[key] = str(type(value).__name__)
        return out

    return {
        "repo_id": repo_id,
        "root": str(root),
        "fps": fps,
        "len": len(dataset),
        "delta_timestamps": delta_timestamps,
        "first_item": item_summary(first),
        "last_item_padding": {
            "action_is_pad": first_to_list(last.get("action_is_pad")),
            f"{camera_key}_is_pad": first_to_list(last.get(f"{camera_key}_is_pad")),
        },
        "first_task": first.get("task"),
        "downloaded_bytes": total_bytes,
    }


def first_to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-key", required=True)
    parser.add_argument("--compare-repo-id")
    parser.add_argument("--local-oxe-name")
    parser.add_argument("--local-oxe-root", default="/mnt/data/oxe")
    parser.add_argument("--cache-root", default="/tmp/lerobot_oxe_probe")
    parser.add_argument("--download-root", default="/tmp/lerobot_stage2_probe")
    parser.add_argument("--probe-runtime", action="store_true")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--pair-offset-steps", type=int, default=5)
    parser.add_argument("--action-chunk-size", type=int, default=10)
    parser.add_argument("--video-backend", default="pyav")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cache_root = Path(args.cache_root)
    download_root = Path(args.download_root)

    report: dict[str, Any] = {
        "primary_repo": summarize_repo_metadata(args.repo_id, cache_root),
    }

    if args.compare_repo_id:
        compare = summarize_repo_metadata(args.compare_repo_id, cache_root)
        report["compare_repo"] = compare
        report["comparison"] = compare_repo_metadata(report["primary_repo"], compare)

    if args.local_oxe_name:
        report["local_oxe"] = summarize_local_oxe_first_episode(
            dataset_name=args.local_oxe_name,
            local_root=Path(args.local_oxe_root),
        )

    if args.probe_runtime:
        report["runtime_probe"] = probe_runtime(
            repo_id=args.repo_id,
            camera_key=args.camera_key,
            episode=args.episode,
            pair_offset_steps=args.pair_offset_steps,
            action_chunk_size=args.action_chunk_size,
            download_root=download_root,
            video_backend=args.video_backend,
        )

    print(json.dumps(_to_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
