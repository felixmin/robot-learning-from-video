"""
LAPA LAM Package

Keep imports lightweight so submodules (e.g. `lam.models.flow`) can be used
without requiring the full training stack (e.g. Lightning).
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "LAMTask",
    "separate_weight_decayable_params",
    "EMACallback",
    "TrainPreviewBufferCallback",
    "ValidationStrategyCallback",
    "LatentActionQuantization",
    "NSVQ",
    "Attention",
    "Transformer",
    # Checkpoint loaders
    "load_lam_task_from_checkpoint",
    "load_lam_model_weights_only",
    "load_lam_encoder_vq_inference_from_checkpoint",
    # Inference
    "LAMEncoderVQInference",
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
    "LAMTask": ("lam.task", "LAMTask"),
    "separate_weight_decayable_params": (
        "lam.task",
        "separate_weight_decayable_params",
    ),
    "EMACallback": ("lam.callbacks", "EMACallback"),
    "TrainPreviewBufferCallback": ("lam.callbacks", "TrainPreviewBufferCallback"),
    "ValidationStrategyCallback": ("lam.callbacks", "ValidationStrategyCallback"),
    "LatentActionQuantization": (
        "lam.models.latent_action_quantization",
        "LatentActionQuantization",
    ),
    "NSVQ": ("lam.models.nsvq", "NSVQ"),
    "Attention": ("lam.models.attention", "Attention"),
    "Transformer": ("lam.models.attention", "Transformer"),
    "load_lam_task_from_checkpoint": (
        "lam.checkpoints",
        "load_lam_task_from_checkpoint",
    ),
    "load_lam_model_weights_only": ("lam.checkpoints", "load_lam_model_weights_only"),
    "load_lam_encoder_vq_inference_from_checkpoint": (
        "lam.checkpoints",
        "load_lam_encoder_vq_inference_from_checkpoint",
    ),
    "LAMEncoderVQInference": ("lam.inference", "LAMEncoderVQInference"),
    "ValidationStrategy": ("lam.validation", "ValidationStrategy"),
    "ValidationCache": ("lam.validation", "ValidationCache"),
    "BasicVisualizationStrategy": ("lam.validation", "BasicVisualizationStrategy"),
    "LatentTransferStrategy": ("lam.validation", "LatentTransferStrategy"),
    "CodebookEmbeddingStrategy": ("lam.validation", "CodebookEmbeddingStrategy"),
    "SequenceExamplesStrategy": ("lam.validation", "SequenceExamplesStrategy"),
    "create_validation_strategies": ("lam.validation", "create_validation_strategies"),
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
