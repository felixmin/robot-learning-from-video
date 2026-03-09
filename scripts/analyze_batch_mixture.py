#!/usr/bin/env python3
"""
Dependency-free mixture analysis from batch_times.jsonl.

Outputs:
- mixture_over_time.svg
- mixture_windows_every_stride.svg
- mixture_summary.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PALETTE = [
    "#0b7285",
    "#1c7ed6",
    "#2b8a3e",
    "#5f3dc4",
    "#e67700",
    "#c2255c",
    "#495057",
    "#6741d9",
    "#2f9e44",
    "#d9480f",
]


def _load_records(path: Path) -> list[dict]:
    recs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {}
    return {str(k): float(v) / total for k, v in counts.items()}


def _top_k_datasets(measured: list[dict], k: int) -> list[str]:
    totals = Counter()
    for r in measured:
        for d, s in _normalize_counts(r.get("dataset_counts", {}) or {}).items():
            totals[d] += float(s)
    return [d for d, _ in totals.most_common(k)]


def _rolling_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values[:]
    w = max(1, int(window))
    out = [0.0] * len(values)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        out[i] = (prefix[hi] - prefix[lo]) / float(max(1, hi - lo))
    return out


def _series_for_categories(
    measured: list[dict], categories: list[str]
) -> dict[str, list[float]]:
    out = {c: [] for c in categories}
    out["other"] = []
    for r in measured:
        shares = _normalize_counts(r.get("dataset_counts", {}) or {})
        used = 0.0
        for c in categories:
            v = float(shares.get(c, 0.0))
            out[c].append(v)
            used += v
        out["other"].append(max(0.0, 1.0 - used))
    return out


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
    )


def _legend_items(categories: list[str]) -> list[tuple[str, str]]:
    items = []
    for i, c in enumerate(categories):
        items.append((c, PALETTE[i % len(PALETTE)]))
    items.append(("other", "#adb5bd"))
    return items


def _write_mixture_over_time_svg(
    measured: list[dict],
    categories: list[str],
    out_path: Path,
    rolling: int,
) -> None:
    width, height = 1700, 520
    margin_l, margin_r, margin_t, margin_b = 70, 270, 40, 55
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    series = _series_for_categories(measured, categories)
    for key in list(series.keys()):
        series[key] = _rolling_average(series[key], rolling)

    keys = categories + ["other"]
    colors = {k: c for k, c in _legend_items(categories)}

    x_step = plot_w / float(max(1, len(measured)))
    bar_w = max(1.0, x_step)

    parts = [_svg_header(width, height)]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
    )
    parts.append(
        f'<text x="{margin_l}" y="24" font-size="16" font-family="monospace">Dataset Mixture Over Time</text>\n'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#444" stroke-width="1"/>\n'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#444" stroke-width="1"/>\n'
    )

    for i in range(len(measured)):
        y_base = margin_t + plot_h
        x = margin_l + i * x_step
        for k in keys:
            frac = max(0.0, min(1.0, float(series[k][i])))
            h = frac * plot_h
            if h <= 0.0:
                continue
            y = y_base - h
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                f'fill="{colors[k]}" stroke="none"/>\n'
            )
            y_base = y

    for tick, yv in [
        (0.0, plot_h),
        (0.25, plot_h * 0.75),
        (0.5, plot_h * 0.5),
        (0.75, plot_h * 0.25),
        (1.0, 0.0),
    ]:
        y = margin_t + yv
        parts.append(
            f'<line x1="{margin_l-5}" y1="{y:.1f}" x2="{margin_l}" y2="{y:.1f}" stroke="#444" stroke-width="1"/>\n'
        )
        parts.append(
            f'<text x="{margin_l-12}" y="{y+4:.1f}" text-anchor="end" font-size="10" font-family="monospace">{tick:.2f}</text>\n'
        )

    n = len(measured)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(frac * max(0, n - 1))
        x = margin_l + frac * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin_t+plot_h}" x2="{x:.1f}" y2="{margin_t+plot_h+5}" stroke="#444" stroke-width="1"/>\n'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{margin_t+plot_h+18}" text-anchor="middle" font-size="10" font-family="monospace">{idx}</text>\n'
        )

    lx = margin_l + plot_w + 20
    ly = margin_t + 15
    for name, color in _legend_items(categories):
        parts.append(
            f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}"/>\n'
        )
        parts.append(
            f'<text x="{lx+18}" y="{ly}" font-size="11" font-family="monospace">{name}</text>\n'
        )
        ly += 18

    parts.append("</svg>\n")
    out_path.write_text("".join(parts), encoding="utf-8")


def _write_windows_svg(
    measured: list[dict],
    categories: list[str],
    out_path: Path,
    window_size: int,
    stride: int,
) -> None:
    starts = list(range(0, max(0, len(measured) - window_size + 1), stride))
    if not starts:
        raise ValueError("Not enough measured batches for requested windows.")

    row_h = 90
    width = 1700
    height = 80 + row_h * len(starts)
    margin_l, margin_r, margin_t = 120, 270, 40
    plot_w = width - margin_l - margin_r
    bar_group_w = plot_w / float(max(1, window_size))
    bar_w = bar_group_w * 0.75

    series = _series_for_categories(measured, categories)
    keys = categories + ["other"]
    colors = {k: c for k, c in _legend_items(categories)}

    parts = [_svg_header(width, height)]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
    )
    parts.append(
        f'<text x="{margin_l}" y="24" font-size="16" font-family="monospace">5-Batch Windows Every {stride} Batches</text>\n'
    )

    for r_i, s in enumerate(starts):
        y0 = margin_t + r_i * row_h
        h = row_h - 20
        parts.append(
            f'<text x="{margin_l-10}" y="{y0 + h/2 + 4:.1f}" text-anchor="end" font-size="10" font-family="monospace">b{s}-{s+window_size-1}</text>\n'
        )
        parts.append(
            f'<line x1="{margin_l}" y1="{y0+h}" x2="{margin_l+plot_w}" y2="{y0+h}" stroke="#444" stroke-width="1"/>\n'
        )
        for b in range(window_size):
            idx = s + b
            x = margin_l + b * bar_group_w + (bar_group_w - bar_w) / 2.0
            y_base = y0 + h
            for k in keys:
                frac = max(0.0, min(1.0, float(series[k][idx])))
                hh = frac * h
                if hh <= 0.0:
                    continue
                y = y_base - hh
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{hh:.2f}" fill="{colors[k]}" stroke="none"/>\n'
                )
                y_base = y

    lx = margin_l + plot_w + 20
    ly = margin_t + 15
    for name, color in _legend_items(categories):
        parts.append(
            f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}"/>\n'
        )
        parts.append(
            f'<text x="{lx+18}" y="{ly}" font-size="11" font-family="monospace">{name}</text>\n'
        )
        ly += 18

    parts.append("</svg>\n")
    out_path.write_text("".join(parts), encoding="utf-8")


def _write_summary(
    measured: list[dict], top_datasets: list[str], out_path: Path
) -> None:
    dom = [r.get("dominant_dataset") for r in measured]
    dom = [d for d in dom if isinstance(d, str) and d]
    switches = sum(1 for i in range(1, len(dom)) if dom[i] != dom[i - 1])
    switch_rate = float(switches) / float(max(1, len(dom) - 1))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Mixture summary\n")
        f.write("================\n")
        f.write(f"measured_batches: {len(measured)}\n")
        f.write(f"unique_dominant_datasets: {len(set(dom))}\n")
        f.write(f"dominant_switch_rate: {switch_rate:.4f}\n")
        f.write(f"top_datasets: {top_datasets}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze batch mixture from benchmark jsonl."
    )
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--rolling", type=int, default=25)
    args = p.parse_args()

    records = _load_records(args.jsonl)
    measured = [r for r in records if not bool(r.get("is_warmup", False))]
    if not measured:
        raise ValueError("No measured (non-warmup) records found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_datasets = _top_k_datasets(measured, max(1, int(args.top_k)))

    _write_mixture_over_time_svg(
        measured=measured,
        categories=top_datasets,
        out_path=args.out_dir / "mixture_over_time.svg",
        rolling=max(1, int(args.rolling)),
    )
    _write_windows_svg(
        measured=measured,
        categories=top_datasets,
        out_path=args.out_dir / "mixture_windows_every_stride.svg",
        window_size=max(1, int(args.window_size)),
        stride=max(1, int(args.stride)),
    )
    _write_summary(
        measured=measured,
        top_datasets=top_datasets,
        out_path=args.out_dir / "mixture_summary.txt",
    )

    print(f"Wrote analysis artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
