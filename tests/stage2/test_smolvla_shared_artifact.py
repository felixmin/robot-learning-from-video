from __future__ import annotations

import torch
import pytest

from stage2.backends.smolvla_shared.artifact import (
    SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION,
    SmolVLASharedArtifactManifest,
    load_smolvla_shared_artifact,
    save_smolvla_shared_artifact,
)


def _manifest(
    *, schema_version: str = SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION
) -> SmolVLASharedArtifactManifest:
    return SmolVLASharedArtifactManifest(
        schema_version=schema_version,
        model_name="dummy",
        torch_dtype="bf16",
        image_size=(384, 384),
        action_dim=32,
        action_chunk_size=50,
        code_seq_len=4,
        latent_vector_dim=128,
        tokenizer_max_length=48,
        pad_language_to="longest",
        system_prompt="You are a robot policy.",
        max_state_dim=32,
        camera_keys=("observation.images.front",),
        flow_hidden_dim=1024,
        flow_steps=8,
        min_period=4e-3,
        max_period=4.0,
        time_beta_alpha=1.5,
        time_beta_beta=1.0,
        source_run_dir="/tmp/run",
        normalization_stats={"action": {"mean": [0.0], "std": [1.0]}},
    )


def test_manifest_validation_rejects_schema_mismatch() -> None:
    with pytest.raises(
        ValueError, match="Unsupported smolvla_shared artifact schema_version"
    ):
        _manifest(schema_version="smolvla_shared.v0").validate()


def test_save_and_load_artifact_roundtrip(tmp_path) -> None:
    path = tmp_path / "artifact.pt"
    core_state_dict = {
        "latent_in_proj.weight": torch.randn(4, 8),
        "latent_in_proj.bias": torch.randn(4),
        "_private_cache.weight": torch.randn(4, 8),
    }
    save_smolvla_shared_artifact(
        path=path, manifest=_manifest(), core_state_dict=core_state_dict
    )

    loaded_manifest, loaded_core = load_smolvla_shared_artifact(path=path)
    assert loaded_manifest.schema_version == SMOLVLA_SHARED_ARTIFACT_SCHEMA_VERSION
    assert loaded_manifest.model_name == "dummy"
    assert loaded_manifest.normalization_stats == {
        "action": {"mean": [0.0], "std": [1.0]}
    }
    assert "_private_cache.weight" not in loaded_core
    expected_keys = {"latent_in_proj.weight", "latent_in_proj.bias"}
    assert set(loaded_core.keys()) == expected_keys
    for key in expected_keys:
        expected = core_state_dict[key]
        assert torch.equal(loaded_core[key], expected)


def test_load_fails_on_invalid_artifact_schema(tmp_path) -> None:
    artifact_path = tmp_path / "bad_artifact.pt"
    manifest = _manifest().to_dict()
    manifest["schema_version"] = "smolvla_shared.v0"
    torch.save(
        {
            "schema_version": "smolvla_shared.v0",
            "manifest": manifest,
            "core_state_dict": {"latent_in_proj.weight": torch.randn(4, 8)},
        },
        artifact_path,
    )

    with pytest.raises(
        ValueError, match="Unsupported smolvla_shared artifact schema_version"
    ):
        load_smolvla_shared_artifact(path=artifact_path)
