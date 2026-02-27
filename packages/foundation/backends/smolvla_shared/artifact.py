from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Mapping

import torch


SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION = "smolvla_shared.v2"
SMOLVLA_SHARED_ARTIFACT_FILENAME = "smolvla_shared_stage2_artifact.pt"


@dataclass(frozen=True)
class SmolVLASharedArtifactManifest:
    schema_version: str
    model_name: str
    torch_dtype: str
    image_size: tuple[int, int]
    action_dim: int
    action_chunk_size: int
    code_seq_len: int
    latent_vector_dim: int
    tokenizer_max_length: int
    pad_language_to: str
    system_prompt: str | None
    max_state_dim: int
    camera_keys: tuple[str, ...] | None
    flow_hidden_dim: int
    flow_steps: int
    min_period: float
    max_period: float
    time_beta_alpha: float
    time_beta_beta: float
    source_backend: str = "smolvla_shared"
    source_training_mode: str | None = None
    source_run_dir: str | None = None
    source_global_step: int | None = None
    normalization_stats: Mapping[str, Mapping[str, Any]] | None = None

    def validate(self) -> None:
        if self.schema_version != SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported smolvla_shared artifact schema_version={self.schema_version!r}; "
                f"expected {SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION!r}"
            )
        if not str(self.model_name).strip():
            raise ValueError("manifest.model_name must be non-empty")
        if not str(self.torch_dtype).strip():
            raise ValueError("manifest.torch_dtype must be non-empty")

        if len(self.image_size) != 2:
            raise ValueError(f"manifest.image_size must have length 2, got {self.image_size!r}")
        if int(self.image_size[0]) <= 0 or int(self.image_size[1]) <= 0:
            raise ValueError(f"manifest.image_size values must be > 0, got {self.image_size!r}")

        if int(self.action_dim) <= 0:
            raise ValueError(f"manifest.action_dim must be > 0, got {self.action_dim}")
        if int(self.action_chunk_size) <= 0:
            raise ValueError(f"manifest.action_chunk_size must be > 0, got {self.action_chunk_size}")
        if int(self.code_seq_len) <= 0:
            raise ValueError(f"manifest.code_seq_len must be > 0, got {self.code_seq_len}")
        if int(self.latent_vector_dim) <= 0:
            raise ValueError(f"manifest.latent_vector_dim must be > 0, got {self.latent_vector_dim}")
        if int(self.tokenizer_max_length) <= 0:
            raise ValueError(
                f"manifest.tokenizer_max_length must be > 0, got {self.tokenizer_max_length}"
            )
        if not str(self.pad_language_to).strip():
            raise ValueError("manifest.pad_language_to must be non-empty")
        if int(self.max_state_dim) <= 0:
            raise ValueError(f"manifest.max_state_dim must be > 0, got {self.max_state_dim}")
        if int(self.flow_hidden_dim) <= 0:
            raise ValueError(f"manifest.flow_hidden_dim must be > 0, got {self.flow_hidden_dim}")
        if int(self.flow_steps) <= 0:
            raise ValueError(f"manifest.flow_steps must be > 0, got {self.flow_steps}")

        if float(self.min_period) <= 0.0:
            raise ValueError(f"manifest.min_period must be > 0, got {self.min_period}")
        if float(self.max_period) <= 0.0:
            raise ValueError(f"manifest.max_period must be > 0, got {self.max_period}")
        if float(self.min_period) >= float(self.max_period):
            raise ValueError(
                f"manifest.min_period must be < manifest.max_period, got {self.min_period} >= {self.max_period}"
            )
        if float(self.time_beta_alpha) <= 0.0:
            raise ValueError(f"manifest.time_beta_alpha must be > 0, got {self.time_beta_alpha}")
        if float(self.time_beta_beta) <= 0.0:
            raise ValueError(f"manifest.time_beta_beta must be > 0, got {self.time_beta_beta}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "model_name": str(self.model_name),
            "torch_dtype": str(self.torch_dtype),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "action_dim": int(self.action_dim),
            "action_chunk_size": int(self.action_chunk_size),
            "code_seq_len": int(self.code_seq_len),
            "latent_vector_dim": int(self.latent_vector_dim),
            "tokenizer_max_length": int(self.tokenizer_max_length),
            "pad_language_to": str(self.pad_language_to),
            "system_prompt": None if self.system_prompt is None else str(self.system_prompt),
            "max_state_dim": int(self.max_state_dim),
            "camera_keys": None if self.camera_keys is None else [str(k) for k in self.camera_keys],
            "flow_hidden_dim": int(self.flow_hidden_dim),
            "flow_steps": int(self.flow_steps),
            "min_period": float(self.min_period),
            "max_period": float(self.max_period),
            "time_beta_alpha": float(self.time_beta_alpha),
            "time_beta_beta": float(self.time_beta_beta),
            "source_backend": str(self.source_backend),
            "source_training_mode": (
                None if self.source_training_mode is None else str(self.source_training_mode)
            ),
            "source_run_dir": None if self.source_run_dir is None else str(self.source_run_dir),
            "source_global_step": (
                None if self.source_global_step is None else int(self.source_global_step)
            ),
            "normalization_stats": self.normalization_stats,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SmolVLASharedArtifactManifest:
        manifest = cls(
            schema_version=str(raw["schema_version"]),
            model_name=str(raw["model_name"]),
            torch_dtype=str(raw["torch_dtype"]),
            image_size=(int(raw["image_size"][0]), int(raw["image_size"][1])),
            action_dim=int(raw["action_dim"]),
            action_chunk_size=int(raw["action_chunk_size"]),
            code_seq_len=int(raw["code_seq_len"]),
            latent_vector_dim=int(raw["latent_vector_dim"]),
            tokenizer_max_length=int(raw["tokenizer_max_length"]),
            pad_language_to=str(raw["pad_language_to"]),
            system_prompt=(None if raw["system_prompt"] is None else str(raw["system_prompt"])),
            max_state_dim=int(raw["max_state_dim"]),
            camera_keys=(
                None
                if raw["camera_keys"] is None
                else tuple(str(x) for x in raw["camera_keys"])
            ),
            flow_hidden_dim=int(raw["flow_hidden_dim"]),
            flow_steps=int(raw["flow_steps"]),
            min_period=float(raw["min_period"]),
            max_period=float(raw["max_period"]),
            time_beta_alpha=float(raw["time_beta_alpha"]),
            time_beta_beta=float(raw["time_beta_beta"]),
            source_backend=str(raw["source_backend"]),
            source_training_mode=(
                None if raw["source_training_mode"] is None else str(raw["source_training_mode"])
            ),
            source_run_dir=(None if raw["source_run_dir"] is None else str(raw["source_run_dir"])),
            source_global_step=(
                None if raw["source_global_step"] is None else int(raw["source_global_step"])
            ),
            normalization_stats=raw["normalization_stats"],
        )
        manifest.validate()
        return manifest


def safe_torch_load(path: Path) -> Any:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(str(path), **load_kwargs)


def _validate_tensor_state_dict(name: str, state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    clean: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} key must be str, got {type(key)}")
        if not torch.is_tensor(value):
            raise TypeError(f"{name}[{key!r}] must be a torch.Tensor, got {type(value)}")
        clean[key] = value
    return clean


def _filter_export_state_dict(core_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Keep stable module parameters only; drop private cache mirrors (keys prefixed with "_").
    return {k: v for k, v in core_state_dict.items() if not k.startswith("_")}


def save_smolvla_shared_artifact(
    *,
    path: Path,
    manifest: SmolVLASharedArtifactManifest,
    core_state_dict: Mapping[str, torch.Tensor],
) -> Path:
    manifest.validate()
    state = _filter_export_state_dict(
        _validate_tensor_state_dict("core_state_dict", core_state_dict),
    )

    payload = {
        "schema_version": manifest.schema_version,
        "manifest": manifest.to_dict(),
        "core_state_dict": {k: v.detach().cpu() for k, v in state.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


def load_smolvla_shared_artifact(
    *,
    path: Path,
) -> tuple[SmolVLASharedArtifactManifest, dict[str, torch.Tensor]]:
    payload = safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Artifact payload must be mapping, got {type(payload)}")
    manifest = SmolVLASharedArtifactManifest.from_dict(payload["manifest"])

    if str(payload["schema_version"]) != manifest.schema_version:
        raise ValueError(
            f"Artifact schema_version mismatch between payload and manifest in {path}: "
            f"{payload['schema_version']!r} vs {manifest.schema_version!r}"
        )
    core_state_dict = _validate_tensor_state_dict("core_state_dict", payload["core_state_dict"])
    return manifest, core_state_dict
