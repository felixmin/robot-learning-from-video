from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from common.filters import matches_filters


STAGE2_METADATA_KEYS = frozenset(
    {
        "dataset_name",
        "dataset_short",
        "episode_id",
        "frame_idx",
        "language",
        "task",
    }
)


def prune_stage2_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if k in STAGE2_METADATA_KEYS}


@dataclass
class Stage2BucketConfig:
    name: str
    filters: dict[str, Any] = field(default_factory=dict)
    max_samples: int = 100
    is_holdout: bool = False

    def matches(self, metadata: dict[str, Any]) -> bool:
        return matches_filters(metadata, self.filters)


class Stage2ValidationCache:
    def __init__(
        self,
        *,
        max_samples: int = 256,
        bucket_name: Optional[str] = None,
        is_holdout: bool = False,
    ) -> None:
        self.max_samples = int(max_samples)
        self.bucket_name = bucket_name
        self.is_holdout = bool(is_holdout)

        self.records: list[dict[str, Any]] = []
        self.all_gt_codes: list[torch.Tensor] = []

        self.fixed_indices: Optional[list[int]] = None
        self.fixed_records: Optional[list[dict[str, Any]]] = None

    def clear(self, *, keep_fixed: bool = True) -> None:
        self.records.clear()
        self.all_gt_codes.clear()
        if not keep_fixed:
            self.fixed_indices = None
            self.fixed_records = None

    def sample_count(self) -> int:
        return len(self.records)

    def add_record(self, record: dict[str, Any]) -> None:
        if self.sample_count() >= self.max_samples:
            return
        rec = dict(record)
        meta = rec.get("metadata")
        if isinstance(meta, dict):
            rec["metadata"] = prune_stage2_metadata(meta)
        else:
            rec["metadata"] = {}
        self.records.append(rec)

    def add_gt_codes_batch(self, codes_batch: Optional[torch.Tensor]) -> None:
        if codes_batch is None:
            return
        if not isinstance(codes_batch, torch.Tensor):
            return
        if codes_batch.ndim == 1:
            codes_batch = codes_batch.unsqueeze(0)
        self.all_gt_codes.append(codes_batch.detach().cpu().to(torch.long))

    def get_all_gt_codes(self) -> Optional[torch.Tensor]:
        if not self.all_gt_codes:
            return None
        return torch.cat(self.all_gt_codes, dim=0)

    def get_records(self) -> list[dict[str, Any]]:
        return self.records

    def get_records_by_filter(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        if not filters:
            return list(self.records)
        out: list[dict[str, Any]] = []
        for rec in self.records:
            meta = rec.get("metadata")
            if isinstance(meta, dict) and matches_filters(meta, filters):
                out.append(rec)
        return out
