from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

from stage2.backends.interfaces import BackendMode, Stage2Batch, LatentOutput, LossOutput
from stage2.policy_module import PolicyLightningModule, PolicyOptimizerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_policy_hlrp" / "src"))
from lerobot_policy_hlrp.policies.hlrp_smolvla_shared.modeling_hlrp_smolvla_shared import (
    HLRPSmolVLASharedPolicy,
)


class _CaptureBackend:
    codebook_size = 8
    code_seq_len = 1

    def __init__(self) -> None:
        self.last_batch: Stage2Batch | None = None
        self.last_mode: BackendMode | None = None

    def setup(self, *, device: torch.device) -> None:
        del device

    def loss_from_batch(self, batch: Stage2Batch, *, mode: BackendMode) -> LossOutput:
        self.last_batch = batch
        self.last_mode = mode
        return LossOutput(loss=torch.tensor(0.0), metrics={"loss": 0.0})

    def latent_from_batch(self, batch: Stage2Batch, *, mode: BackendMode) -> LatentOutput:
        del batch, mode
        return LatentOutput()


@dataclass
class _DummyCodeProvider:
    codebook_size: int = 8
    code_seq_len: int = 1
    codebook_dim: int = 2


def test_stage2_stage3_adapter_parity_for_core_inputs() -> None:
    stats = {
        "action": {"mean": [1.0, 2.0], "std": [2.0, 2.0]},
        "observation.state": {"mean": [1.0, -1.0], "std": [2.0, 4.0]},
    }

    backend = _CaptureBackend()
    module = PolicyLightningModule(
        backend=backend,
        code_provider=_DummyCodeProvider(),
        backend_mode=BackendMode.ACTIONS,
        normalization_stats=stats,
        optimizer=PolicyOptimizerConfig(lr=1e-4, weight_decay=0.0),
    )

    stage2_input = Stage2Batch(
        image_streams={"primary": torch.randint(0, 255, (2, 2, 3, 8, 8), dtype=torch.uint8)},
        image_padding_masks={"primary": torch.ones((2, 2), dtype=torch.bool)},
        task_text=["pick", "place"],
        state=torch.tensor([[[3.0, 3.0]], [[1.0, -1.0]]], dtype=torch.float32),
        target_actions=torch.tensor([[[3.0, 6.0]], [[1.0, 2.0]]], dtype=torch.float32),
        action_is_pad=torch.zeros((2, 1), dtype=torch.bool),
    )
    module._loss_and_targets_from_batch(stage2_input)

    assert backend.last_batch is not None
    stage2_batch = backend.last_batch

    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy._image_keys = ["observation.images.rgb"]
    policy._action_dim = 2
    policy.config = SimpleNamespace(chunk_size=1, max_action_dim=2)
    policy.normalization_stats = stats
    policy.core = torch.nn.Linear(1, 1)

    lerobot_batch = {
        "observation.images.rgb": stage2_batch.image_streams["primary"],
        "observation.images.rgb_is_pad": torch.zeros((2, 1), dtype=torch.bool),
        "task": ["pick", "place"],
        "observation.state": torch.tensor([[3.0, 3.0], [1.0, -1.0]], dtype=torch.float32),
        "action": torch.tensor([[3.0, 6.0], [1.0, 2.0]], dtype=torch.float32),
        "action_is_pad": torch.zeros((2, 1), dtype=torch.bool),
    }
    stage3_batch = HLRPSmolVLASharedPolicy._to_stage2_batch(
        policy,
        lerobot_batch,
        require_action_is_pad=True,
        require_image_padding_masks=True,
        conditioning_step_index=-1,
    )
    stage3_action_target = HLRPSmolVLASharedPolicy._extract_action_target(policy, lerobot_batch)

    assert torch.equal(
        stage2_batch.image_streams["primary"][:, -1, ...],
        stage3_batch.image_streams["observation.images.rgb"],
    )
    assert torch.equal(stage2_batch.action_is_pad, stage3_batch.action_is_pad)
    assert torch.allclose(stage2_batch.state.squeeze(1), stage3_batch.state)
    assert torch.allclose(stage2_batch.target_actions.squeeze(1), stage3_action_target.squeeze(1))


def test_stage3_policy_accepts_actions_id_pad_alias() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy._image_keys = ["observation.images.rgb"]
    policy._action_dim = 2
    policy.config = SimpleNamespace(chunk_size=1, max_action_dim=2)
    policy.normalization_stats = None
    policy.core = torch.nn.Linear(1, 1)

    batch = {
        "observation.images.rgb": torch.randint(0, 255, (2, 1, 8, 8, 3), dtype=torch.uint8),
        "observation.images.rgb_is_pad": torch.zeros((2, 1), dtype=torch.bool),
        "task": ["pick", "place"],
        "observation.state": torch.zeros((2, 2), dtype=torch.float32),
        "actions_id_pad": torch.tensor([[False], [True]], dtype=torch.bool),
    }
    out = HLRPSmolVLASharedPolicy._to_stage2_batch(
        policy,
        batch,
        require_action_is_pad=True,
        require_image_padding_masks=True,
        conditioning_step_index=-1,
    )
    assert torch.equal(out.action_is_pad, batch["actions_id_pad"])


def test_stage3_policy_rejects_conflicting_mask_aliases() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy._image_keys = ["observation.images.rgb"]
    policy._action_dim = 2
    policy.config = SimpleNamespace(chunk_size=1, max_action_dim=2)
    policy.normalization_stats = None
    policy.core = torch.nn.Linear(1, 1)

    batch = {
        "observation.images.rgb": torch.randint(0, 255, (1, 1, 8, 8, 3), dtype=torch.uint8),
        "observation.images.rgb_is_pad": torch.zeros((1, 1), dtype=torch.bool),
        "task": ["pick"],
        "observation.state": torch.zeros((1, 2), dtype=torch.float32),
        "action_is_pad": torch.tensor([[False]], dtype=torch.bool),
        "actions_id_pad": torch.tensor([[True]], dtype=torch.bool),
    }
    try:
        HLRPSmolVLASharedPolicy._to_stage2_batch(
            policy,
            batch,
            require_action_is_pad=True,
            require_image_padding_masks=True,
            conditioning_step_index=-1,
        )
    except ValueError as e:
        assert "Conflicting action pad masks" in str(e)
    else:
        raise AssertionError("Expected ValueError for conflicting action pad masks")


def test_stage3_policy_inference_batch_allows_missing_action_mask() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy._image_keys = ["observation.images.rgb"]
    policy._action_dim = 2
    policy.config = SimpleNamespace(chunk_size=1, max_action_dim=2)
    policy.normalization_stats = None
    policy.core = torch.nn.Linear(1, 1)

    batch = {
        "observation.images.rgb": torch.randint(0, 255, (2, 1, 8, 8, 3), dtype=torch.uint8),
        "observation.images.rgb_is_pad": torch.zeros((2, 1), dtype=torch.bool),
        "task": ["pick", "place"],
        "observation.state": torch.zeros((2, 2), dtype=torch.float32),
    }
    out = HLRPSmolVLASharedPolicy._to_stage2_batch(
        policy,
        batch,
        require_action_is_pad=False,
        require_image_padding_masks=False,
        conditioning_step_index=-1,
    )
    assert out.action_is_pad is None


def test_stage3_policy_inference_batch_allows_missing_image_is_pad() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy._image_keys = ["observation.images.rgb"]
    policy._action_dim = 2
    policy.config = SimpleNamespace(chunk_size=1, max_action_dim=2)
    policy.normalization_stats = None
    policy.core = torch.nn.Linear(1, 1)

    batch = {
        "observation.images.rgb": torch.randint(0, 255, (2, 1, 8, 8, 3), dtype=torch.uint8),
        "task": ["pick", "place"],
        "observation.state": torch.zeros((2, 2), dtype=torch.float32),
    }
    out = HLRPSmolVLASharedPolicy._to_stage2_batch(
        policy,
        batch,
        require_action_is_pad=False,
        require_image_padding_masks=False,
        conditioning_step_index=-1,
    )
    mask = out.image_padding_masks["observation.images.rgb"]
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.ones((2,), dtype=torch.bool))


def test_stage3_action_supervision_mask_prefix_ratio() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_subset_ratio=0.4, action_subset_key="index")
    policy.dataset_meta = SimpleNamespace(total_frames=10, total_episodes=5)
    policy._action_supervision_threshold = None

    batch = {"index": torch.tensor([0, 3, 4, 9], dtype=torch.long)}
    mask = HLRPSmolVLASharedPolicy._action_supervision_mask(
        policy,
        batch,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, torch.tensor([True, True, False, False]))


def test_stage3_action_supervision_mask_ratio_one_selects_all() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_subset_ratio=1.0, action_subset_key="index")
    policy.dataset_meta = None
    policy._action_supervision_threshold = None

    batch = {"index": torch.tensor([100, 200], dtype=torch.long)}
    mask = HLRPSmolVLASharedPolicy._action_supervision_mask(
        policy,
        batch,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, torch.tensor([True, True]))


def test_stage3_zero_loss_uses_trainable_param_when_first_param_is_frozen() -> None:
    policy = object.__new__(HLRPSmolVLASharedPolicy)
    torch.nn.Module.__init__(policy)
    policy.core = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    for param in policy.core[0].parameters():
        param.requires_grad_(False)

    loss = HLRPSmolVLASharedPolicy._zero_loss(policy)

    assert loss.requires_grad
    loss.backward()
    assert policy.core[0].weight.grad is None
    assert policy.core[1].weight.grad is not None
