from __future__ import annotations

import gc
import logging
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

from foundation.validation_cache import Stage2BucketConfig, Stage2ValidationCache
from foundation.validation_checks import CHECK_TYPES, Stage2ValidationCheck

logger = logging.getLogger("foundation.training")


class Stage2ValidationPipelineCallback(Callback):
    def __init__(
        self,
        *,
        checks_config: dict[str, Any],
        bucket_configs: dict[str, Any] | None = None,
        max_cached_samples: int = 256,
        num_fixed_samples: int = 8,
        run_gc_after_validation: bool = True,
    ) -> None:
        super().__init__()
        self.num_fixed_samples = int(num_fixed_samples)
        self.run_gc_after_validation = bool(run_gc_after_validation)

        self.global_cache = Stage2ValidationCache(max_samples=int(max_cached_samples))

        self.bucket_configs: dict[str, Stage2BucketConfig] = {}
        if bucket_configs:
            for name, cfg in bucket_configs.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"Bucket {name} config must be a dict")
                self.bucket_configs[str(name)] = Stage2BucketConfig(
                    name=str(name),
                    filters=dict(cfg.get("filters", {})),
                    max_samples=int(cfg.get("max_samples", max_cached_samples)),
                    is_holdout=bool(cfg.get("is_holdout", False)),
                )

        self.bucket_caches: dict[str, Stage2ValidationCache] = {}
        for name, bucket in self.bucket_configs.items():
            self.bucket_caches[name] = Stage2ValidationCache(
                max_samples=int(bucket.max_samples),
                bucket_name=name,
                is_holdout=bool(bucket.is_holdout),
            )

        self.checks = self._create_checks(checks_config or {})
        self._validate_check_bindings()

    def _create_checks(self, checks_config: dict[str, Any]) -> list[Stage2ValidationCheck]:
        checks: list[Stage2ValidationCheck] = []
        for instance_name, raw_cfg in checks_config.items():
            cfg = dict(raw_cfg or {})
            check_type = str(cfg.get("type", instance_name))
            check_cls = CHECK_TYPES.get(check_type)
            if check_cls is None:
                raise ValueError(f"Unknown Stage 2 validation check type {check_type} (instance {instance_name})")

            kwargs = {k: v for k, v in cfg.items() if k != "type"}
            kwargs["name"] = str(instance_name)

            if check_type == "latent_flow_decode" and bool(kwargs.get("enabled", True)):
                ckpt = str(kwargs.get("laq_checkpoint_path", "")).strip()
                if not ckpt:
                    raise ValueError(
                        f"Check {instance_name} ({check_type}) requires non-empty laq_checkpoint_path"
                    )

            checks.append(check_cls(**kwargs))
        return checks

    def _validate_check_bindings(self) -> None:
        known = set(self.bucket_caches.keys())
        for check in self.checks:
            for bucket_name in check.buckets:
                if bucket_name not in known:
                    raise ValueError(
                        f"Check {check.name} references unknown bucket {bucket_name}. Known buckets: {sorted(known)}"
                    )

    @staticmethod
    def _select_fixed_records(cache: Stage2ValidationCache, num_samples: int) -> None:
        records = cache.get_records()
        if not records or num_samples <= 0:
            return
        if len(records) < num_samples:
            cache.fixed_indices = list(range(len(records)))
            cache.fixed_records = list(records)
            return

        groups: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            dname = str(meta.get("dataset_name", "unknown"))
            groups.setdefault(dname, []).append(i)

        selected: list[int] = []
        group_names = list(groups.keys())
        if group_names:
            per_group = max(1, num_samples // len(group_names))
            for name in group_names:
                idxs = groups[name]
                perm = torch.randperm(len(idxs)).tolist()
                for j in perm[:per_group]:
                    if len(selected) >= num_samples:
                        break
                    selected.append(idxs[j])

        if len(selected) < num_samples:
            remaining = [i for i in range(len(records)) if i not in set(selected)]
            perm = torch.randperm(len(remaining)).tolist() if remaining else []
            for j in perm:
                if len(selected) >= num_samples:
                    break
                selected.append(remaining[j])

        cache.fixed_indices = selected
        cache.fixed_records = [records[i] for i in selected]

    @staticmethod
    def _result_is_no_output(result: Any) -> tuple[bool, str]:
        if not isinstance(result, dict):
            raise TypeError("Validation check must return dict result")
        if "_produced" not in result:
            raise KeyError("Validation check result missing required '_produced' field")
        produced = int(result.get("_produced", 0))
        if produced > 0:
            return False, ""
        reason = result.get("_reason")
        return True, str(reason) if reason is not None else "no outputs produced"

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.global_cache.clear(keep_fixed=True)
        for cache in self.bucket_caches.values():
            cache.clear(keep_fixed=True)

        reset_fn = getattr(pl_module, "reset_val_batch_payload_queue", None)
        if callable(reset_fn):
            reset_fn()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        consume_fn = getattr(pl_module, "consume_next_val_batch_payload", None)
        if not callable(consume_fn):
            return

        payload = consume_fn()
        if not isinstance(payload, dict):
            return

        records = payload.get("records")
        if not isinstance(records, list):
            return

        for rec in records:
            if not isinstance(rec, dict):
                continue
            self.global_cache.add_record(rec)

            gt_codes = rec.get("gt_codes")
            if isinstance(gt_codes, list):
                try:
                    code_t = torch.tensor(gt_codes, dtype=torch.long).unsqueeze(0)
                    self.global_cache.add_gt_codes_batch(code_t)
                except Exception:
                    pass

            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            for bucket_name, bucket_cfg in self.bucket_configs.items():
                if not bucket_cfg.matches(meta):
                    continue
                bucket_cache = self.bucket_caches[bucket_name]
                bucket_cache.add_record(rec)
                if isinstance(gt_codes, list):
                    try:
                        code_t = torch.tensor(gt_codes, dtype=torch.long).unsqueeze(0)
                        bucket_cache.add_gt_codes_batch(code_t)
                    except Exception:
                        pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.global_cache.fixed_records is None:
            self._select_fixed_records(self.global_cache, self.num_fixed_samples)
            for cache in self.bucket_caches.values():
                self._select_fixed_records(cache, min(4, self.num_fixed_samples))

        due = 0
        ran = 0
        skipped = 0
        soft_failed = 0

        for check in self.checks:
            if not check.enabled:
                continue

            if not check.buckets:
                due += 1
                can_run, reason = check.can_run(self.global_cache)
                if not can_run:
                    skipped += 1
                    logger.warning("[Stage2Validation] skip %s: %s", check.name, reason)
                    continue
                try:
                    result = check.run(self.global_cache, pl_module, trainer, metric_suffix="")
                    no_output, reason = self._result_is_no_output(result)
                    if no_output:
                        skipped += 1
                        logger.warning("[Stage2Validation] skip %s: %s", check.name, reason)
                    else:
                        ran += 1
                except Exception as e:
                    soft_failed += 1
                    logger.warning("[Stage2Validation] %s failed: %s", check.name, e)
                continue

            for bucket_name in check.buckets:
                due += 1
                cache = self.bucket_caches[bucket_name]
                suffix = f"_{bucket_name}"
                if cache.is_holdout:
                    suffix += "_holdout"

                can_run, reason = check.can_run(cache)
                if not can_run:
                    skipped += 1
                    logger.warning("[Stage2Validation] skip %s%s: %s", check.name, suffix, reason)
                    continue

                try:
                    result = check.run(cache, pl_module, trainer, metric_suffix=suffix)
                    no_output, reason = self._result_is_no_output(result)
                    if no_output:
                        skipped += 1
                        logger.warning("[Stage2Validation] skip %s%s: %s", check.name, suffix, reason)
                    else:
                        ran += 1
                except Exception as e:
                    soft_failed += 1
                    logger.warning("[Stage2Validation] %s%s failed: %s", check.name, suffix, e)

        logger.info(
            "[Stage2Validation] due=%d ran=%d skipped=%d soft_failed=%d",
            due,
            ran,
            skipped,
            soft_failed,
        )

        if self.run_gc_after_validation:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
