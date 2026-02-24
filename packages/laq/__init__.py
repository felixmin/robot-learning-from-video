"""
LAPA LAQ Package

Keep imports lightweight so submodules (e.g. `laq.models.flow`) can be used
without requiring the full training stack (e.g. Lightning).
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "LAQTask",
    "separate_weight_decayable_params",
    "EMACallback",
    "TrainPreviewBufferCallback",
    "ValidationStrategyCallback",
    "LatentActionQuantization",
    "NSVQ",
    "Attention",
    "Transformer",
    # Validation
    "ValidationStrategy",
    "ValidationCache",
    "BasicVisualizationStrategy",
    "LatentTransferStrategy",
    "CodebookEmbeddingStrategy",
    "SequenceExamplesStrategy",
    "create_validation_strategies",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LAQTask": ("laq.task", "LAQTask"),
    "separate_weight_decayable_params": ("laq.task", "separate_weight_decayable_params"),
    "EMACallback": ("laq.callbacks", "EMACallback"),
    "TrainPreviewBufferCallback": ("laq.callbacks", "TrainPreviewBufferCallback"),
    "ValidationStrategyCallback": ("laq.callbacks", "ValidationStrategyCallback"),
    "LatentActionQuantization": ("laq.models.latent_action_quantization", "LatentActionQuantization"),
    "NSVQ": ("laq.models.nsvq", "NSVQ"),
    "Attention": ("laq.models.attention", "Attention"),
    "Transformer": ("laq.models.attention", "Transformer"),
    "ValidationStrategy": ("laq.validation", "ValidationStrategy"),
    "ValidationCache": ("laq.validation", "ValidationCache"),
    "BasicVisualizationStrategy": ("laq.validation", "BasicVisualizationStrategy"),
    "LatentTransferStrategy": ("laq.validation", "LatentTransferStrategy"),
    "CodebookEmbeddingStrategy": ("laq.validation", "CodebookEmbeddingStrategy"),
    "SequenceExamplesStrategy": ("laq.validation", "SequenceExamplesStrategy"),
    "create_validation_strategies": ("laq.validation", "create_validation_strategies"),
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
