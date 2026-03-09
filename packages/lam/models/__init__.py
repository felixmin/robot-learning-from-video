"""
LAM Model Components

Avoid eager imports so lightweight submodules (e.g. `lam.models.flow`) do not
require optional heavy dependencies (e.g. `transformers`).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "NSVQ",
    "Attention",
    "Transformer",
    "ContinuousPositionBias",
    "PEG",
    "LatentActionQuantization",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NSVQ": ("lam.models.nsvq", "NSVQ"),
    "Attention": ("lam.models.attention", "Attention"),
    "Transformer": ("lam.models.attention", "Transformer"),
    "ContinuousPositionBias": ("lam.models.attention", "ContinuousPositionBias"),
    "PEG": ("lam.models.attention", "PEG"),
    "LatentActionQuantization": (
        "lam.models.latent_action_quantization",
        "LatentActionQuantization",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
