from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _resolve_libero_benchmark_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "LIBERO package is not importable; cannot bootstrap LIBERO config"
        )

    package_dir = Path(spec.origin).resolve().parent
    benchmark_root = package_dir / "libero"
    if not (benchmark_root / "bddl_files").exists():
        benchmark_root = package_dir
    if not (benchmark_root / "bddl_files").exists():
        raise RuntimeError(
            f"Could not locate LIBERO benchmark assets under {package_dir}"
        )
    return benchmark_root


def ensure_libero_config(*, env: dict[str, str], logger: Any) -> None:
    config_path_raw = env.get("LIBERO_CONFIG_PATH")
    if not config_path_raw:
        return

    config_root = Path(config_path_raw)
    config_file = (
        config_root
        if config_root.suffix.lower() in {".yaml", ".yml"}
        else config_root / "config.yaml"
    )
    if config_file.exists():
        return

    benchmark_root = _resolve_libero_benchmark_root()
    if config_file.parent.name == "libero_config":
        cache_root = config_file.parent.parent
    else:
        cache_root = config_file.parent
    datasets_dir = cache_root / "libero" / "datasets"
    assets_dir = cache_root / "libero" / "assets"

    config_file.parent.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    def _yaml_str(value: Path) -> str:
        return json.dumps(str(value))

    config_body = "\n".join(
        [
            f"benchmark_root: {_yaml_str(benchmark_root)}",
            f"bddl_files: {_yaml_str(benchmark_root / 'bddl_files')}",
            f"init_states: {_yaml_str(benchmark_root / 'init_files')}",
            f"datasets: {_yaml_str(datasets_dir)}",
            f"assets: {_yaml_str(assets_dir)}",
            "",
        ]
    )
    config_file.write_text(config_body, encoding="utf-8")
    logger.info("Bootstrapped LIBERO config: %s", config_file)
