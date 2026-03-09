"""Create Lightning DataModules from the Hydra `cfg.data` schema."""

from __future__ import annotations

from typing import Any, Dict

try:
    from omegaconf import DictConfig, OmegaConf
except (
    ModuleNotFoundError
):  # pragma: no cover - used only in lightweight unit-test envs
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if DictConfig is not None and isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(out, dict):
            raise TypeError(f"Expected DictConfig -> dict, got {type(out)}")
        return out
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"Expected DictConfig or dict, got {type(cfg)}")


def create_datamodule(cfg_data: Any):
    """
    Create a Lightning DataModule from `cfg.data`.

    Expected schema:
      data:
        backend: lerobot_v3
        preprocess: {image_size: int, return_metadata: bool}
        loader: {batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int|null}
        dataset: {...}
        adapter: {...}
    """
    data = _to_dict(cfg_data)

    backend = data["backend"]
    loader = data["loader"]
    dataset = data["dataset"]

    if backend == "lerobot_v3":
        from common.lerobot_v3_data import LeRobotV3DataModule

        adapter = data["adapter"]["lerobot_v3"]
        lerobot = dataset["lerobot"]
        return LeRobotV3DataModule(
            sources=list(lerobot["sources"]),
            request=data["request"],
            loader=loader,
            adapter=adapter,
            output_format=str(data["output_format"]),
        )

    raise ValueError(f"Only data.backend='lerobot_v3' is supported, got {backend!r}")
