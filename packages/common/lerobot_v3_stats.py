from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def load_source_stats(meta: Any) -> dict[str, dict[str, np.ndarray]]:
    stats = getattr(meta, "stats", None)
    if stats is None:
        return {}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, entry in stats.items():
        out[str(key)] = {
            str(stat_key): np.asarray(stat_value)
            for stat_key, stat_value in entry.items()
        }
    return out


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weights, got shape {tuple(weights.shape)}")
    if weights.size == 0:
        raise ValueError("Expected at least one source weight")
    if np.any(weights < 0):
        raise ValueError("Source weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Source weights must sum to a positive value")
    return weights / total


def _to_flat_array(value: np.ndarray | Any, *, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if out.ndim == 0:
        return out.reshape(1)
    return out.reshape(-1)


def _pad_last_dim(
    array: np.ndarray, *, target_dim: int, fill_value: float
) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 1:
        raise ValueError(f"Expected 1D stats array, got shape {tuple(array.shape)}")
    if int(array.shape[0]) > int(target_dim):
        raise ValueError(
            f"Stats dim {int(array.shape[0])} exceeds target_dim {int(target_dim)}"
        )
    padded = np.full((int(target_dim),), fill_value, dtype=array.dtype)
    valid = np.zeros((int(target_dim),), dtype=np.bool_)
    padded[: int(array.shape[0])] = array
    valid[: int(array.shape[0])] = True
    return padded, valid


def merge_weighted_stats(
    stats_by_source: list[dict[str, dict[str, np.ndarray]]],
    source_weights: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if not stats_by_source:
        return {}
    if len(stats_by_source) != int(source_weights.shape[0]):
        raise ValueError("stats_by_source and source_weights length mismatch")

    weights = normalize_weights(np.asarray(source_weights, dtype=np.float64))
    merged: dict[str, dict[str, np.ndarray]] = {}
    feature_keys = {key for stats in stats_by_source for key in stats}
    for feature_key in feature_keys:
        present = [
            (stats[feature_key], weights[idx])
            for idx, stats in enumerate(stats_by_source)
            if feature_key in stats
        ]
        if not present:
            continue
        local_weights = np.asarray([w for _, w in present], dtype=np.float64)
        means = [
            _to_flat_array(entry["mean"], dtype=np.float64) for entry, _ in present
        ]
        stds = [_to_flat_array(entry["std"], dtype=np.float64) for entry, _ in present]
        mins = [_to_flat_array(entry["min"], dtype=np.float64) for entry, _ in present]
        maxs = [_to_flat_array(entry["max"], dtype=np.float64) for entry, _ in present]
        counts = [
            _to_flat_array(
                entry.get("count", np.asarray([0], dtype=np.int64)), dtype=np.int64
            )
            for entry, _ in present
        ]
        feature_dim = max(int(arr.shape[0]) for arr in means)

        mean_arrays: list[np.ndarray] = []
        std_arrays: list[np.ndarray] = []
        min_arrays: list[np.ndarray] = []
        max_arrays: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        for mu, sigma, min_value, max_value in zip(
            means, stds, mins, maxs, strict=True
        ):
            mu_pad, valid = _pad_last_dim(mu, target_dim=feature_dim, fill_value=0.0)
            sigma_pad, _ = _pad_last_dim(sigma, target_dim=feature_dim, fill_value=0.0)
            min_pad, _ = _pad_last_dim(
                min_value, target_dim=feature_dim, fill_value=np.inf
            )
            max_pad, _ = _pad_last_dim(
                max_value, target_dim=feature_dim, fill_value=-np.inf
            )
            mean_arrays.append(mu_pad)
            std_arrays.append(sigma_pad)
            min_arrays.append(min_pad)
            max_arrays.append(max_pad)
            valid_masks.append(valid)

        valid_matrix = np.stack(valid_masks, axis=0)
        weight_matrix = valid_matrix.astype(np.float64) * local_weights[:, None]
        denom = weight_matrix.sum(axis=0)
        if np.any(denom <= 0.0):
            raise ValueError(
                f"Feature {feature_key!r} has no valid stats coverage on some dimensions"
            )
        normalized_weight_matrix = weight_matrix / denom[None, :]

        mean = np.zeros((feature_dim,), dtype=np.float64)
        for weight_row, mu in zip(normalized_weight_matrix, mean_arrays, strict=True):
            mean = mean + (weight_row * mu)

        variance = np.zeros((feature_dim,), dtype=np.float64)
        for weight_row, mu, sigma in zip(
            normalized_weight_matrix, mean_arrays, std_arrays, strict=True
        ):
            variance = variance + (weight_row * ((sigma**2) + ((mu - mean) ** 2)))

        feature_stats: dict[str, np.ndarray] = {
            "mean": mean,
            "std": np.sqrt(variance),
            "min": np.minimum.reduce(np.stack(min_arrays, axis=0)),
            "max": np.maximum.reduce(np.stack(max_arrays, axis=0)),
        }
        if all(int(count.shape[0]) == 1 for count in counts):
            feature_stats["count"] = np.asarray(
                [sum(int(count[0]) for count in counts)],
                dtype=np.int64,
            )
        else:
            count_arrays = [
                _pad_last_dim(
                    count.astype(np.int64), target_dim=feature_dim, fill_value=0
                )[0]
                for count in counts
            ]
            feature_stats["count"] = np.sum(np.stack(count_arrays, axis=0), axis=0)

        quantile_keys = {
            stat_key
            for entry, _ in present
            for stat_key in entry
            if stat_key.startswith("q") and stat_key[1:].isdigit()
        }
        for quantile_key in quantile_keys:
            if all(quantile_key in entry for entry, _ in present):
                values = [
                    _pad_last_dim(
                        _to_flat_array(entry[quantile_key], dtype=np.float64),
                        target_dim=feature_dim,
                        fill_value=0.0,
                    )[0]
                    for entry, _ in present
                ]
                q_value = np.zeros((feature_dim,), dtype=np.float64)
                for weight_row, value in zip(
                    normalized_weight_matrix, values, strict=True
                ):
                    q_value = q_value + (weight_row * value)
                feature_stats[quantile_key] = q_value

        merged[feature_key] = feature_stats
    return merged


def build_run_normalization_stats(
    sources: Sequence[Any],
    *,
    weights_mode: str,
) -> dict[str, dict[str, np.ndarray]]:
    if weights_mode != "explicit":
        raise NotImplementedError(weights_mode)
    stats_by_source: list[dict[str, dict[str, np.ndarray]]] = []
    weights: list[float] = []
    for source in sources:
        stats_by_source.append(load_source_stats(source.meta))
        weights.append(float(getattr(source, "weight")))
    return merge_weighted_stats(stats_by_source, np.asarray(weights, dtype=np.float64))
