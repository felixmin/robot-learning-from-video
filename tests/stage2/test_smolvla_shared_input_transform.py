from __future__ import annotations

import torch
import pytest

from stage2.backends.interfaces import Stage2Batch
from stage2.backends.smolvla_shared.input_transform import (
    ImageTransformConfig,
    LanguageTransformConfig,
    normalize_vector_mean_std,
    prepare_image_inputs,
    prepare_language_inputs,
    resolve_action_pad_field,
    resolve_action_pad_mask,
    to_action_chunk,
    unnormalize_vector_mean_std,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_texts: list[str] | None = None

    def __call__(
        self,
        texts,
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ):
        assert return_tensors == "pt"
        assert truncation is True
        assert padding in {"longest", "max_length"}
        self.last_texts = [str(x) for x in texts]
        b = len(texts)
        seq_len = min(8, max_length)
        return {
            "input_ids": torch.ones((b, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((b, seq_len), dtype=torch.long),
        }


def test_prepare_language_inputs_uses_newline_and_system_prompt() -> None:
    tokenizer = _FakeTokenizer()
    batch = Stage2Batch(task_text=["pick cube", "place cube"])
    tokens, mask = prepare_language_inputs(
        batch=batch,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        config=LanguageTransformConfig(
            system_prompt="You are a robot policy.",
            tokenizer_max_length=16,
            pad_language_to="longest",
        ),
    )
    assert tokens.shape == (2, 8)
    assert mask.shape == (2, 8)
    assert tokenizer.last_texts is not None
    assert tokenizer.last_texts[0].startswith("You are a robot policy.\n")
    assert tokenizer.last_texts[0].endswith("\n")


def test_prepare_image_inputs_supports_multi_camera_and_empty_camera() -> None:
    batch = Stage2Batch(
        image_streams={
            "cam_front": torch.randint(0, 255, (2, 2, 16, 16, 3), dtype=torch.uint8),
            "cam_wrist": torch.randint(0, 255, (2, 1, 16, 16, 3), dtype=torch.uint8),
        },
        image_padding_masks={
            "cam_front": torch.tensor(
                [[False, False], [False, True]], dtype=torch.bool
            ),
            "cam_wrist": torch.tensor([[False], [False]], dtype=torch.bool),
        },
    )
    images, masks = prepare_image_inputs(
        batch=batch,
        device=torch.device("cpu"),
        config=ImageTransformConfig(
            image_size=(32, 32),
            camera_keys=("cam_front", "cam_wrist", "cam_missing"),
            empty_cameras=1,
            normalize_to_neg_one_to_one=True,
        ),
    )
    assert len(images) == 3
    assert len(masks) == 3
    assert tuple(images[0].shape) == (2, 3, 32, 32)
    assert tuple(masks[0].shape) == (2,)
    # last camera is synthetic empty camera
    assert torch.all(masks[2] == 0)


def test_prepare_image_inputs_default_camera_order_preserves_stream_insertion() -> None:
    cam_b = torch.zeros((1, 1, 8, 8, 3), dtype=torch.uint8)
    cam_a = torch.full((1, 1, 8, 8, 3), 255, dtype=torch.uint8)
    batch = Stage2Batch(
        image_streams={
            "cam_b": cam_b,
            "cam_a": cam_a,
        },
        image_padding_masks={
            "cam_b": torch.tensor([[False]], dtype=torch.bool),
            "cam_a": torch.tensor([[False]], dtype=torch.bool),
        },
    )
    images, _masks = prepare_image_inputs(
        batch=batch,
        device=torch.device("cpu"),
        config=ImageTransformConfig(
            image_size=(8, 8),
            normalize_to_neg_one_to_one=True,
            camera_keys=None,
            empty_cameras=0,
        ),
    )
    assert len(images) == 2
    # cam_b is inserted first and remains first; zero image maps to -1 after normalization.
    assert float(images[0].mean().item()) < float(images[1].mean().item())


def test_prepare_language_inputs_requires_task_text_without_pretokenized_inputs() -> (
    None
):
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError, match="task_text"):
        prepare_language_inputs(
            batch=Stage2Batch(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            config=LanguageTransformConfig(
                system_prompt=None,
                tokenizer_max_length=16,
                pad_language_to="longest",
            ),
        )


def test_prepare_image_inputs_requires_image_streams() -> None:
    with pytest.raises(ValueError, match="image_streams"):
        prepare_image_inputs(
            batch=Stage2Batch(),
            device=torch.device("cpu"),
            config=ImageTransformConfig(
                image_size=(32, 32),
                normalize_to_neg_one_to_one=True,
                camera_keys=("cam_front",),
                empty_cameras=0,
            ),
        )


def test_resolve_action_pad_mask_requires_exact_chunk_size() -> None:
    mask = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
    with pytest.raises(ValueError, match="time mismatch"):
        resolve_action_pad_mask(
            action_is_pad=mask,
            batch_size=2,
            chunk_size=4,
            device=torch.device("cpu"),
        )


def test_to_action_chunk_requires_exact_time_dim() -> None:
    actions = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="Expected action tensor"):
        to_action_chunk(actions=actions, chunk_size=2)


def test_to_action_chunk_2d_only_allowed_for_chunk_size_1() -> None:
    actions = torch.randn(2, 4)
    with pytest.raises(ValueError, match="chunk_size>1"):
        to_action_chunk(actions=actions, chunk_size=3)


def test_resolve_action_pad_field_accepts_alias() -> None:
    batch = {"actions_id_pad": torch.tensor([[False, True]], dtype=torch.bool)}
    out = resolve_action_pad_field(
        batch=batch,
        action_is_pad_key="action_is_pad",
        actions_id_pad_key="actions_id_pad",
    )
    assert torch.equal(out, batch["actions_id_pad"])


def test_resolve_action_pad_field_rejects_conflict() -> None:
    batch = {
        "action_is_pad": torch.tensor([[False, True]], dtype=torch.bool),
        "actions_id_pad": torch.tensor([[False, False]], dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="Conflicting action pad masks"):
        resolve_action_pad_field(
            batch=batch,
            action_is_pad_key="action_is_pad",
            actions_id_pad_key="actions_id_pad",
        )


def test_mean_std_normalization_handles_padded_tail_dims() -> None:
    x = torch.zeros((2, 4, 6), dtype=torch.float32)
    stats = {
        "action": {
            "mean": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "std": torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
        }
    }
    n = normalize_vector_mean_std(value=x, stats=stats, key_candidates=["action"])
    u = unnormalize_vector_mean_std(value=n, stats=stats, key_candidates=["action"])
    assert tuple(n.shape) == (2, 4, 6)
    # tail dims are identity-normalized because stats are shorter than padded model dim
    assert torch.allclose(n[..., 3:], x[..., 3:])
    assert torch.allclose(u, x)


def test_mean_std_normalization_pads_value_to_stats_dim() -> None:
    x = torch.tensor([[[3.0, 5.0]]], dtype=torch.float32)
    stats = {
        "state": {
            "mean": torch.tensor([1.0, 3.0, 7.0], dtype=torch.float32),
            "std": torch.tensor([2.0, 2.0, 5.0], dtype=torch.float32),
        }
    }
    n = normalize_vector_mean_std(value=x, stats=stats, key_candidates=["state"])
    u = unnormalize_vector_mean_std(value=n, stats=stats, key_candidates=["state"])
    assert tuple(n.shape) == (1, 1, 3)
    assert torch.allclose(n, torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float32))
    assert torch.allclose(u, torch.tensor([[[3.0, 5.0, 7.0]]], dtype=torch.float32))
