from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig

from common.cache_env import configure_cache_env, resolve_cache_dir
from common.unified_logging import resolve_runs_dir, setup_unified_logging


def setup_run_context(*, cfg: Any, workspace_root: Path, logger_name: str):
    runs_dir = None
    try:
        if HydraConfig.initialized():
            runs_dir = Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        runs_dir = None

    if runs_dir is None:
        runs_dir = resolve_runs_dir(
            logging_root_dir=cfg.logging.root_dir,
            logging_runs_dir=cfg.logging.runs_dir,
            workspace_root=workspace_root,
            experiment_name=cfg.experiment.name,
        )

    logger, output_dir = setup_unified_logging(
        runs_dir=runs_dir,
        job_id=cfg.logging.job_id,
        log_level=str(cfg.logging.level),
        logger_name=logger_name,
    )

    cache_dir = resolve_cache_dir(cfg=cfg, workspace_root=workspace_root)
    if cache_dir is not None:
        configure_cache_env(cache_dir=cache_dir, logger=logger)

    return logger, output_dir
