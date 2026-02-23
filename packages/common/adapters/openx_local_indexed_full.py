"""
Indexed-full local OpenX adapter.

This module provides a random-access local OXE pipeline with:
- persistent per-split episode index arrays
- map-style dataset loading by (episode_id, t)
- hierarchical dataset/episode/t sampler
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import tarfile
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, Sampler

from common.adapters.oxe_shared import (
    OXE_DATASETS,
    OXEDatasetConfig,
    resolve_oxe_dataset_config,
    resolve_oxe_dataset_key,
)

from .openx_local import (
    _decode_image_to_tensor,
    _extract_action_step,
    _extract_image,
    _extract_instruction,
    _extract_state,
    _fallback_dataset_config,
    _parse_train_split,
)


INDEX_SCHEMA_VERSION = "openx_local_indexed_full_v1"
logger = logging.getLogger(__name__)


@dataclass
class OpenXLocalEpisodeIndexBundle:
    index_path: str
    key: str
    datasets: List[str]
    dataset_weights: np.ndarray
    dataset_offsets: np.ndarray
    dataset_configs: List[OXEDatasetConfig]
    shard_paths: List[str]
    episode_dataset_ids: np.ndarray
    episode_shard_ids: np.ndarray
    episode_offsets: np.ndarray
    episode_sizes: np.ndarray
    episode_num_steps: np.ndarray
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        return int(self.episode_num_steps.shape[0])


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _iter_episode_records_from_shard(
    shard_path: str,
) -> Iterator[tuple[int, int, int]]:
    """Yield (offset_data, size, num_steps) for each valid episode."""
    try:
        with tarfile.open(shard_path, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".data.pickle"):
                    continue
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    continue
                try:
                    episode = pickle.load(fileobj)
                except Exception:
                    continue
                steps = episode.get("steps") if isinstance(episode, dict) else None
                if not isinstance(steps, list):
                    continue
                n_steps = len(steps)
                if n_steps <= 0:
                    continue
                yield int(member.offset_data), int(member.size), int(n_steps)
    except (tarfile.ReadError, OSError):
        return


def _scan_shard_records_to_tmp(
    *,
    task_idx: int,
    entry_id: int,
    dataset_name: str,
    shard_path: str,
    shard_id: int,
    tmp_dir: str,
) -> Dict[str, Any]:
    episode_offsets = array("Q")
    episode_sizes = array("Q")
    episode_num_steps = array("I")

    for offset_data, size, n_steps in _iter_episode_records_from_shard(shard_path):
        episode_offsets.append(offset_data)
        episode_sizes.append(size)
        episode_num_steps.append(n_steps)

    stem = f"task_{task_idx:06d}_shard_{shard_id:06d}"
    offsets_path = os.path.join(tmp_dir, f"{stem}_offsets.npy")
    sizes_path = os.path.join(tmp_dir, f"{stem}_sizes.npy")
    steps_path = os.path.join(tmp_dir, f"{stem}_num_steps.npy")

    np.save(offsets_path, np.frombuffer(episode_offsets, dtype=np.uint64))
    np.save(sizes_path, np.frombuffer(episode_sizes, dtype=np.uint64))
    np.save(steps_path, np.frombuffer(episode_num_steps, dtype=np.uint32))

    return {
        "task_idx": int(task_idx),
        "entry_id": int(entry_id),
        "dataset_name": str(dataset_name),
        "shard_id": int(shard_id),
        "episode_count": int(len(episode_offsets)),
        "offsets_path": offsets_path,
        "sizes_path": sizes_path,
        "steps_path": steps_path,
    }


def _required_files(index_root: Path) -> List[Path]:
    return [
        index_root / "meta.json",
        index_root / "episode_dataset_ids.npy",
        index_root / "episode_shard_ids.npy",
        index_root / "episode_offsets.npy",
        index_root / "episode_sizes.npy",
        index_root / "episode_num_steps.npy",
    ]


def _is_complete_index(index_root: Path) -> bool:
    return all(p.exists() for p in _required_files(index_root))


def _index_fingerprint(
    *,
    root: str,
    dataset_entries: List[Dict[str, Any]],
    max_shards_per_dataset: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root_path = Path(root)
    for entry in dataset_entries:
        dataset_name = str(entry["name"])
        dataset_dir = root_path / dataset_name
        if not dataset_dir.exists():
            continue
        shards = sorted(dataset_dir.glob("*.tar"))
        if max_shards_per_dataset is not None:
            shards = shards[:max_shards_per_dataset]
        for shard in shards:
            st = shard.stat()
            rows.append(
                {
                    "dataset": dataset_name,
                    "shard": str(shard.resolve()),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
    return rows


def _index_key(
    *,
    root: str,
    split_key: str,
    dataset_entries: List[Dict[str, Any]],
    max_shards_per_dataset: Optional[int],
    fingerprint: List[Dict[str, Any]],
) -> str:
    payload = {
        "schema": INDEX_SCHEMA_VERSION,
        "root": str(Path(root).resolve()),
        "split_key": split_key,
        "max_shards_per_dataset": max_shards_per_dataset,
        "datasets": [
            {
                "name": str(entry["name"]),
                "weight": float(entry.get("weight", 1.0)),
                "offset": int(entry["pair_offset_steps"]),
                "split": str(entry[split_key]),
            }
            for entry in dataset_entries
        ],
        "fingerprint": fingerprint,
    }
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def build_openx_local_episode_index(
    *,
    root: str,
    dataset_entries: List[Dict[str, Any]],
    split_key: str,
    max_shards_per_dataset: Optional[int],
    index_workers: int,
    index_root: str,
    key: str,
    fingerprint: List[Dict[str, Any]],
) -> None:
    out_dir = Path(index_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_path = Path(root)

    shard_paths: List[str] = []
    shard_id_by_path: Dict[str, int] = {}
    entry_metas: List[Dict[str, Any]] = []
    shard_tasks: List[Dict[str, Any]] = []
    shard_counts_by_dataset: Dict[str, int] = {}

    for entry in dataset_entries:
        dataset_name = str(entry["name"])
        split = str(entry[split_key])
        offset = int(entry["pair_offset_steps"])
        weight = float(entry.get("weight", 1.0))

        dataset_dir = root_path / dataset_name
        if not dataset_dir.exists():
            continue

        shards = sorted(dataset_dir.glob("*.tar"))
        if max_shards_per_dataset is not None:
            shards = shards[:max_shards_per_dataset]
        if not shards:
            continue

        entry_id = len(entry_metas)
        entry_metas.append(
            {
                "entry_id": entry_id,
                "dataset_name": dataset_name,
                "split": split,
                "offset": offset,
                "weight": weight,
            }
        )

        for shard in shards:
            shard_abs = str(shard.resolve())
            if shard_abs not in shard_id_by_path:
                shard_id_by_path[shard_abs] = len(shard_paths)
                shard_paths.append(shard_abs)
            shard_counts_by_dataset[dataset_name] = (
                shard_counts_by_dataset.get(dataset_name, 0) + 1
            )
            shard_tasks.append(
                {
                    "task_idx": len(shard_tasks),
                    "entry_id": entry_id,
                    "dataset_name": dataset_name,
                    "shard_path": shard_abs,
                    "shard_id": int(shard_id_by_path[shard_abs]),
                }
            )

    worker_count = int(index_workers)
    if worker_count <= 0:
        worker_count = os.cpu_count() or 1
    worker_count = max(1, worker_count)

    logger.info(
        "→ Building OpenX local episode index: split=%s key=%s datasets=%d shards=%d workers=%d dir=%s",
        split_key,
        key,
        len(entry_metas),
        len(shard_tasks),
        worker_count,
        str(out_dir),
    )
    for meta in entry_metas:
        ds_name = str(meta["dataset_name"])
        logger.info(
            "  - Dataset index start: %s split=%s offset=%d shards=%d",
            ds_name,
            str(meta["split"]),
            int(meta["offset"]),
            int(shard_counts_by_dataset.get(ds_name, 0)),
        )

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shard_results: List[Dict[str, Any]] = []
    try:
        if shard_tasks:
            if worker_count == 1:
                total_tasks = len(shard_tasks)
                for i, task in enumerate(shard_tasks, start=1):
                    logger.info(
                        "  - Shard index start [%d/%d]: dataset=%s shard=%s",
                        i,
                        total_tasks,
                        str(task["dataset_name"]),
                        Path(str(task["shard_path"])).name,
                    )
                    result = _scan_shard_records_to_tmp(tmp_dir=str(tmp_dir), **task)
                    logger.info(
                        "  - Shard index done  [%d/%d]: dataset=%s shard=%s episodes=%d",
                        i,
                        total_tasks,
                        str(result["dataset_name"]),
                        Path(str(task["shard_path"])).name,
                        int(result["episode_count"]),
                    )
                    shard_results.append(result)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as ex:
                    future_to_task: Dict[Any, Dict[str, Any]] = {}
                    total_tasks = len(shard_tasks)
                    for i, task in enumerate(shard_tasks, start=1):
                        logger.info(
                            "  - Shard index start [%d/%d]: dataset=%s shard=%s",
                            i,
                            total_tasks,
                            str(task["dataset_name"]),
                            Path(str(task["shard_path"])).name,
                        )
                        fut = ex.submit(
                            _scan_shard_records_to_tmp, tmp_dir=str(tmp_dir), **task
                        )
                        future_to_task[fut] = task
                    done_count = 0
                    for fut in as_completed(future_to_task):
                        task = future_to_task[fut]
                        result = fut.result()
                        done_count += 1
                        logger.info(
                            "  - Shard index done  [%d/%d]: dataset=%s shard=%s episodes=%d",
                            done_count,
                            total_tasks,
                            str(result["dataset_name"]),
                            Path(str(task["shard_path"])).name,
                            int(result["episode_count"]),
                        )
                        shard_results.append(result)
        else:
            logger.warning("No shard tasks found for split=%s (key=%s)", split_key, key)

        episode_totals = [0 for _ in entry_metas]
        for result in shard_results:
            episode_totals[int(result["entry_id"])] += int(result["episode_count"])

        selected_meta_by_entry: Dict[int, Dict[str, Any]] = {}
        selected_metas: List[Dict[str, Any]] = []
        total_selected = 0
        for entry in entry_metas:
            entry_id = int(entry["entry_id"])
            total_eps = int(episode_totals[entry_id])
            start, end = _parse_train_split(str(entry["split"]), total_eps)
            selected_count = max(0, end - start)
            if selected_count <= 0:
                continue
            dataset_id = len(selected_metas)
            meta = {
                "dataset_id": dataset_id,
                "entry_id": entry_id,
                "dataset_name": str(entry["dataset_name"]),
                "weight": float(entry["weight"]),
                "offset": int(entry["offset"]),
                "split_start": int(start),
                "split_end": int(end),
                "selected_count": int(selected_count),
                "total_count": int(total_eps),
            }
            selected_metas.append(meta)
            selected_meta_by_entry[entry_id] = meta
            total_selected += selected_count
            logger.info(
                "✓ Dataset index done: %s total_episodes=%d selected_episodes=%d split=%s",
                str(entry["dataset_name"]),
                total_eps,
                selected_count,
                str(entry["split"]),
            )

        if total_selected <= 0:
            np.save(out_dir / "episode_dataset_ids.npy", np.empty((0,), dtype=np.uint32))
            np.save(out_dir / "episode_shard_ids.npy", np.empty((0,), dtype=np.uint32))
            np.save(out_dir / "episode_offsets.npy", np.empty((0,), dtype=np.uint64))
            np.save(out_dir / "episode_sizes.npy", np.empty((0,), dtype=np.uint64))
            np.save(out_dir / "episode_num_steps.npy", np.empty((0,), dtype=np.uint32))
        else:
            episode_dataset_ids_mm = open_memmap(
                out_dir / "episode_dataset_ids.npy",
                mode="w+",
                dtype=np.uint32,
                shape=(total_selected,),
            )
            episode_shard_ids_mm = open_memmap(
                out_dir / "episode_shard_ids.npy",
                mode="w+",
                dtype=np.uint32,
                shape=(total_selected,),
            )
            episode_offsets_mm = open_memmap(
                out_dir / "episode_offsets.npy",
                mode="w+",
                dtype=np.uint64,
                shape=(total_selected,),
            )
            episode_sizes_mm = open_memmap(
                out_dir / "episode_sizes.npy",
                mode="w+",
                dtype=np.uint64,
                shape=(total_selected,),
            )
            episode_num_steps_mm = open_memmap(
                out_dir / "episode_num_steps.npy",
                mode="w+",
                dtype=np.uint32,
                shape=(total_selected,),
            )

            entry_seen = [0 for _ in entry_metas]
            cursor = 0
            for result in sorted(shard_results, key=lambda x: int(x["task_idx"])):
                entry_id = int(result["entry_id"])
                n_ep = int(result["episode_count"])
                if n_ep <= 0:
                    continue

                selected_meta = selected_meta_by_entry.get(entry_id)
                local_start = int(entry_seen[entry_id])
                local_end = local_start + n_ep

                if selected_meta is not None:
                    split_start = int(selected_meta["split_start"])
                    split_end = int(selected_meta["split_end"])
                    keep_start = max(local_start, split_start)
                    keep_end = min(local_end, split_end)
                    if keep_end > keep_start:
                        keep_from = keep_start - local_start
                        keep_to = keep_end - local_start
                        keep_n = int(keep_to - keep_from)

                        offsets = np.load(result["offsets_path"], mmap_mode="r")
                        sizes = np.load(result["sizes_path"], mmap_mode="r")
                        steps = np.load(result["steps_path"], mmap_mode="r")

                        ds_id = int(selected_meta["dataset_id"])
                        shard_id = int(result["shard_id"])

                        episode_dataset_ids_mm[cursor : cursor + keep_n] = ds_id
                        episode_shard_ids_mm[cursor : cursor + keep_n] = shard_id
                        episode_offsets_mm[cursor : cursor + keep_n] = offsets[keep_from:keep_to]
                        episode_sizes_mm[cursor : cursor + keep_n] = sizes[keep_from:keep_to]
                        episode_num_steps_mm[cursor : cursor + keep_n] = steps[keep_from:keep_to]
                        cursor += keep_n

                entry_seen[entry_id] = local_end

            if cursor != total_selected:
                raise RuntimeError(
                    f"Indexed-full merge mismatch: wrote {cursor} episodes, expected {total_selected}"
                )

            episode_dataset_ids_mm.flush()
            episode_shard_ids_mm.flush()
            episode_offsets_mm.flush()
            episode_sizes_mm.flush()
            episode_num_steps_mm.flush()

        meta = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "root": str(root_path.resolve()),
            "split_key": split_key,
            "max_shards_per_dataset": max_shards_per_dataset,
            "index_workers": int(worker_count),
            "fingerprint": fingerprint,
            "shard_paths": shard_paths,
            "datasets": [m["dataset_name"] for m in selected_metas],
            "dataset_weights": [float(m["weight"]) for m in selected_metas],
            "dataset_offsets": [int(m["offset"]) for m in selected_metas],
            "dataset_selected_episodes": [int(m["selected_count"]) for m in selected_metas],
            "dataset_total_episodes": [int(m["total_count"]) for m in selected_metas],
            "num_episodes": int(total_selected),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(
            "✓ Indexed-full episode index ready: split=%s key=%s episodes=%d dir=%s",
            split_key,
            key,
            int(total_selected),
            str(out_dir),
        )
    finally:
        for result in shard_results:
            for key_name in ("offsets_path", "sizes_path", "steps_path"):
                p = result.get(key_name)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def load_openx_local_episode_index(index_root: str) -> OpenXLocalEpisodeIndexBundle:
    root = Path(index_root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))

    datasets = [str(x) for x in meta.get("datasets", [])]
    dataset_weights = np.asarray(meta.get("dataset_weights", []), dtype=np.float64)
    dataset_offsets = np.asarray(meta.get("dataset_offsets", []), dtype=np.int64)
    dataset_configs = []
    for name in datasets:
        resolved_cfg = resolve_oxe_dataset_config(name)
        if resolved_cfg is not None:
            dataset_configs.append(resolved_cfg)
            resolved_key = resolve_oxe_dataset_key(name)
            if resolved_key is not None and resolved_key != name:
                logger.info("Resolved dataset alias: %s -> %s", name, resolved_key)
        else:
            dataset_configs.append(_fallback_dataset_config(name))

    return OpenXLocalEpisodeIndexBundle(
        index_path=str(root),
        key=str(meta.get("key", "")),
        datasets=datasets,
        dataset_weights=dataset_weights,
        dataset_offsets=dataset_offsets,
        dataset_configs=dataset_configs,
        shard_paths=[str(x) for x in meta.get("shard_paths", [])],
        episode_dataset_ids=np.load(root / "episode_dataset_ids.npy", mmap_mode="r"),
        episode_shard_ids=np.load(root / "episode_shard_ids.npy", mmap_mode="r"),
        episode_offsets=np.load(root / "episode_offsets.npy", mmap_mode="r"),
        episode_sizes=np.load(root / "episode_sizes.npy", mmap_mode="r"),
        episode_num_steps=np.load(root / "episode_num_steps.npy", mmap_mode="r"),
        metadata=meta,
    )


def prepare_openx_local_episode_index(
    *,
    root: str,
    dataset_entries: List[Dict[str, Any]],
    split_key: str,
    max_shards_per_dataset: Optional[int],
    index_workers: int,
    index_dir: str,
    rebuild: bool,
) -> OpenXLocalEpisodeIndexBundle:
    fingerprint = _index_fingerprint(
        root=root,
        dataset_entries=dataset_entries,
        max_shards_per_dataset=max_shards_per_dataset,
    )
    key = _index_key(
        root=root,
        split_key=split_key,
        dataset_entries=dataset_entries,
        max_shards_per_dataset=max_shards_per_dataset,
        fingerprint=fingerprint,
    )
    index_root = Path(index_dir).expanduser().resolve() / key

    index_complete = _is_complete_index(index_root)
    if rebuild or not index_complete:
        reason = "rebuild=true" if rebuild else "missing_or_incomplete_cache"
        logger.info(
            "→ Indexed-full index build required: split=%s key=%s reason=%s dir=%s",
            split_key,
            key,
            reason,
            str(index_root),
        )
        build_openx_local_episode_index(
            root=root,
            dataset_entries=dataset_entries,
            split_key=split_key,
            max_shards_per_dataset=max_shards_per_dataset,
            index_workers=index_workers,
            index_root=str(index_root),
            key=key,
            fingerprint=fingerprint,
        )
    else:
        logger.info(
            "✓ Reusing indexed-full episode index: split=%s key=%s dir=%s",
            split_key,
            key,
            str(index_root),
        )
    bundle = load_openx_local_episode_index(str(index_root))
    logger.info(
        "✓ Loaded indexed-full episode index: split=%s key=%s datasets=%d episodes=%d",
        split_key,
        key,
        len(bundle.datasets),
        len(bundle),
    )
    return bundle


def _load_episode_from_offset(
    fileobj: BinaryIO, offset_data: int, size: int
) -> Optional[Dict[str, Any]]:
    try:
        fileobj.seek(offset_data)
        payload = fileobj.read(size)
    except Exception:
        return None
    try:
        episode = pickle.loads(payload)
    except Exception:
        return None
    if isinstance(episode, dict) and isinstance(episode.get("steps"), list):
        return episode
    return None


class OpenXLocalIndexedPairMapDataset(Dataset):
    """Map-style indexed-full dataset. Sampler must provide (episode_id, t)."""

    def __init__(
        self,
        *,
        index: OpenXLocalEpisodeIndexBundle,
        image_size: int,
        return_metadata: bool,
        max_open_shards: int = 16,
        max_sample_attempts: int = 24,
    ) -> None:
        self.index = index
        self.image_size = int(image_size)
        self.return_metadata = bool(return_metadata)
        self.max_open_shards = max(1, int(max_open_shards))
        self.max_sample_attempts = max(1, int(max_sample_attempts))
        self._open_files: Dict[int, BinaryIO] = {}
        self._lru_shards: List[int] = []

        self._dataset_offsets = np.asarray(self.index.dataset_offsets, dtype=np.int64)
        self._episode_dataset_ids = np.asarray(self.index.episode_dataset_ids, dtype=np.int64)
        self._episode_num_steps = np.asarray(self.index.episode_num_steps, dtype=np.int64)
        self._episode_offsets = self._dataset_offsets[self._episode_dataset_ids]
        self._episode_max_t = self._episode_num_steps - self._episode_offsets
        self._valid_episode_ids = np.nonzero(self._episode_max_t > 0)[0].astype(
            np.int64, copy=False
        )
        if self._valid_episode_ids.size == 0:
            raise ValueError("Indexed-full dataset has no valid episodes for configured offsets.")

        self._max_action_dim = max((cfg.action_dim for cfg in self.index.dataset_configs), default=0)
        self._max_state_dim = max((cfg.state_dim for cfg in self.index.dataset_configs), default=0)

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_open_files"] = {}
        state["_lru_shards"] = []
        return state

    def __del__(self) -> None:
        for fileobj in self._open_files.values():
            try:
                fileobj.close()
            except Exception:
                pass
        self._open_files.clear()
        self._lru_shards.clear()

    def _touch_lru(self, shard_id: int) -> None:
        if shard_id in self._lru_shards:
            self._lru_shards.remove(shard_id)
        self._lru_shards.append(shard_id)

    def _get_shard_file(self, shard_id: int) -> BinaryIO:
        fileobj = self._open_files.get(shard_id)
        if fileobj is not None:
            self._touch_lru(shard_id)
            return fileobj

        shard_path = self.index.shard_paths[shard_id]
        fileobj = open(shard_path, "rb")
        self._open_files[shard_id] = fileobj
        self._touch_lru(shard_id)

        while len(self._open_files) > self.max_open_shards:
            evict_id = self._lru_shards.pop(0)
            evict_file = self._open_files.pop(evict_id, None)
            if evict_file is not None:
                try:
                    evict_file.close()
                except Exception:
                    pass
        return fileobj

    def _max_t_for_episode(self, episode_id: int) -> int:
        return int(self._episode_max_t[episode_id])

    def _fallback_episode_and_t(
        self, episode_id: int, t: int, attempt: int
    ) -> tuple[int, int]:
        seed = (
            int(episode_id) * 1_000_003
            + int(t) * 9_176
            + int(attempt) * 101
            + int(self.image_size) * 17
            + len(self.index.datasets) * 13
        ) & 0xFFFFFFFF
        rng = random.Random(seed)
        if attempt < 4:
            max_t = self._max_t_for_episode(episode_id)
            if max_t > 0:
                return episode_id, rng.randrange(max_t)
        new_episode = int(
            self._valid_episode_ids[rng.randrange(int(self._valid_episode_ids.shape[0]))]
        )
        max_t = self._max_t_for_episode(new_episode)
        return new_episode, rng.randrange(max_t)

    def _build_by_episode_and_t(self, episode_id: int, t: int) -> Optional[Any]:
        dataset_id = int(self.index.episode_dataset_ids[episode_id])
        offset = int(self.index.dataset_offsets[dataset_id])
        max_t = self._max_t_for_episode(episode_id)
        if max_t <= 0 or t < 0 or t >= max_t:
            return None

        shard_id = int(self.index.episode_shard_ids[episode_id])
        offset_data = int(self.index.episode_offsets[episode_id])
        size = int(self.index.episode_sizes[episode_id])
        fileobj = self._get_shard_file(shard_id)
        episode = _load_episode_from_offset(fileobj, offset_data, size)
        if episode is None:
            return None

        steps = episode.get("steps")
        if not isinstance(steps, list):
            return None
        step_t = steps[t]
        step_h = steps[t + offset]
        if not isinstance(step_t, dict) or not isinstance(step_h, dict):
            return None

        obs_t = step_t.get("observation", {})
        obs_h = step_h.get("observation", {})
        if not isinstance(obs_t, dict) or not isinstance(obs_h, dict):
            return None

        cfg = self.index.dataset_configs[dataset_id]
        raw_img_t = _extract_image(obs_t, cfg)
        raw_img_h = _extract_image(obs_h, cfg)
        if raw_img_t is None or raw_img_h is None:
            return None

        try:
            img_t = _decode_image_to_tensor(raw_img_t, self.image_size)
            img_h = _decode_image_to_tensor(raw_img_h, self.image_size)
        except Exception:
            return None

        frames = torch.stack([img_t, img_h], dim=0).permute(1, 0, 2, 3)
        if not self.return_metadata:
            return frames

        action = np.zeros(self._max_action_dim, dtype=np.float32)
        for j in range(t, t + offset):
            action += _extract_action_step(steps[j], cfg, self._max_action_dim)
        initial_state = _extract_state(obs_t, cfg, self._max_state_dim)
        language = _extract_instruction(step_t, obs_t, cfg)
        shard_path = self.index.shard_paths[shard_id]
        episode_ref = f"{self.index.datasets[dataset_id]}:{Path(shard_path).name}@{offset_data}"

        return {
            "frames": frames,
            "episode_id": episode_ref,
            "frame_idx": int(t),
            "dataset_name": cfg.name,
            "dataset_type": self.index.datasets[dataset_id],
            "language": language,
            "offset": int(offset),
            "action": action,
            "initial_state": initial_state,
        }

    def __getitem__(self, idx: object) -> Any:
        if not (isinstance(idx, tuple) and len(idx) == 2):
            raise TypeError(
                "OpenXLocalIndexedPairMapDataset expects sampler indices as (episode_id, t). "
                f"Got: {type(idx).__name__} value={idx!r}"
            )
        episode_id = int(idx[0])
        t = int(idx[1])
        if episode_id < 0 or episode_id >= len(self.index):
            raise IndexError(f"episode_id out of range: {episode_id}")

        cur_episode = episode_id
        cur_t = t
        for attempt in range(self.max_sample_attempts):
            sample = self._build_by_episode_and_t(cur_episode, cur_t)
            if sample is not None:
                return sample
            cur_episode, cur_t = self._fallback_episode_and_t(cur_episode, cur_t, attempt)

        raise RuntimeError(
            f"Failed to build sample after {self.max_sample_attempts} attempts "
            f"(episode_id={episode_id}, t={t})"
        )


class OpenXLocalIndexedEpisodePairSampler(Sampler[tuple[int, int]]):
    """Hierarchical sampler: dataset -> episode (mass) -> timestep t."""

    def __init__(
        self,
        *,
        index: OpenXLocalEpisodeIndexBundle,
        pairs_per_episode: Optional[int],
        weights_by_size: bool,
        num_samples: Optional[int],
        seed: int,
        epoch: int,
        resample_each_epoch: bool,
    ) -> None:
        self.index = index
        self.pairs_per_episode = (
            int(pairs_per_episode)
            if pairs_per_episode is not None and int(pairs_per_episode) > 0
            else None
        )
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.resample_each_epoch = bool(resample_each_epoch)
        self._external_epoch_set = False
        self._iter_invocations = 0

        self.dataset_ids = np.asarray(index.episode_dataset_ids, dtype=np.int64)
        dataset_offsets = np.asarray(index.dataset_offsets, dtype=np.int64)
        max_t = (
            np.asarray(index.episode_num_steps, dtype=np.int64)
            - dataset_offsets[self.dataset_ids]
        )
        self.max_t = max_t

        n_datasets = len(index.datasets)
        self.episode_ids_by_dataset: List[np.ndarray] = []
        self.episode_mass_prefix_by_dataset: List[np.ndarray] = []
        dataset_mass = np.zeros((n_datasets,), dtype=np.float64)

        for dataset_id in range(n_datasets):
            mask = (self.dataset_ids == dataset_id) & (self.max_t > 0)
            episode_ids = np.nonzero(mask)[0].astype(np.int64, copy=False)
            if episode_ids.size == 0:
                self.episode_ids_by_dataset.append(np.empty((0,), dtype=np.int64))
                self.episode_mass_prefix_by_dataset.append(
                    np.empty((0,), dtype=np.float64)
                )
                continue
            masses = self.max_t[episode_ids].astype(np.float64, copy=False)
            if self.pairs_per_episode is not None:
                masses = np.minimum(masses, float(self.pairs_per_episode))
            masses = np.maximum(masses, 0.0)
            if masses.sum() <= 0.0:
                self.episode_ids_by_dataset.append(np.empty((0,), dtype=np.int64))
                self.episode_mass_prefix_by_dataset.append(
                    np.empty((0,), dtype=np.float64)
                )
                continue
            self.episode_ids_by_dataset.append(episode_ids)
            self.episode_mass_prefix_by_dataset.append(np.cumsum(masses, dtype=np.float64))
            dataset_mass[dataset_id] = float(masses.sum())

        if weights_by_size:
            base = dataset_mass.copy()
        else:
            base = np.asarray(index.dataset_weights, dtype=np.float64)
            if base.shape[0] != n_datasets:
                raise ValueError("dataset_weights length must match indexed datasets")
            base = np.where(dataset_mass > 0.0, np.maximum(base, 0.0), 0.0)

        if base.sum() <= 0.0:
            raise ValueError("No sampleable datasets for indexed_full mode")

        self.dataset_mass = dataset_mass
        self.dataset_probabilities = base / base.sum()
        self.valid_dataset_ids = np.nonzero(self.dataset_probabilities > 0.0)[0].astype(
            np.int64
        )
        self.valid_dataset_probs = self.dataset_probabilities[self.valid_dataset_ids]
        self.valid_dataset_probs = self.valid_dataset_probs / self.valid_dataset_probs.sum()

        if num_samples is None:
            inferred = int(round(float(dataset_mass.sum())))
            self.num_samples = max(1, inferred)
        else:
            self.num_samples = max(1, int(num_samples))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._external_epoch_set = True

    def _epoch_term(self) -> int:
        if not self.resample_each_epoch:
            return 0
        if self._external_epoch_set:
            return self.epoch
        term = int(self._iter_invocations)
        self._iter_invocations += 1
        return term

    def _episode_subset(
        self,
        *,
        episode_id: int,
        max_t: int,
        cap: int,
        epoch_term: int,
        cache: Dict[int, np.ndarray],
    ) -> np.ndarray:
        cached = cache.get(episode_id)
        if cached is not None:
            return cached
        seed = (self.seed + episode_id * 100_003 + epoch_term * 10_000_019) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        k = min(int(cap), int(max_t))
        chosen = rng.choice(max_t, size=k, replace=False).astype(np.int64, copy=False)
        chosen.sort()
        cache[episode_id] = chosen
        return chosen

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch_term = self._epoch_term()
        rng = np.random.default_rng(self.seed + epoch_term * 1_000_003)
        subset_cache: Dict[int, np.ndarray] = {}

        for _ in range(self.num_samples):
            dataset_id = int(rng.choice(self.valid_dataset_ids, p=self.valid_dataset_probs))
            episode_ids = self.episode_ids_by_dataset[dataset_id]
            prefix = self.episode_mass_prefix_by_dataset[dataset_id]
            if episode_ids.size == 0 or prefix.size == 0:
                continue

            u = float(rng.random()) * float(prefix[-1])
            local_idx = int(np.searchsorted(prefix, u, side="right"))
            if local_idx >= episode_ids.size:
                local_idx = episode_ids.size - 1
            episode_id = int(episode_ids[local_idx])
            max_t = int(self.max_t[episode_id])
            if max_t <= 0:
                continue

            if self.pairs_per_episode is None or self.pairs_per_episode >= max_t:
                t = int(rng.integers(0, max_t))
            else:
                subset = self._episode_subset(
                    episode_id=episode_id,
                    max_t=max_t,
                    cap=int(self.pairs_per_episode),
                    epoch_term=epoch_term,
                    cache=subset_cache,
                )
                t = int(subset[rng.integers(0, subset.shape[0])])
            yield (episode_id, t)
