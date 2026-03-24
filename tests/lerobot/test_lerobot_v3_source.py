from __future__ import annotations

from types import SimpleNamespace

import torch

from common.lerobot_v3_source import LeRobotSingleSource

from tests.helpers.lerobot_v3_fixtures import make_test_meta, make_test_request


class _FakeLeRobotDataset:
    init_kwargs = None
    last_instance = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = dict(kwargs)
        type(self).last_instance = self
        self.meta = SimpleNamespace(
            info={
                "features": {
                    "observation.images.image": {"dtype": "video"},
                    "observation.images.wrist": {"dtype": "video"},
                    "observation.state": {"dtype": "float32"},
                    "action": {"dtype": "float32"},
                    "actions": {"dtype": "float32"},
                }
            },
            video_keys=["observation.images.image", "observation.images.wrist"],
        )

    def __getitem__(self, index: int):
        del index
        return {
            "observation.images.image": torch.tensor(
                [
                    [
                        [[0.0, 1.0], [0.5, 0.25]],
                        [[1.0, 0.0], [0.25, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]],
                    ],
                    [
                        [[0.25, 0.75], [0.1, 0.2]],
                        [[0.0, 0.5], [0.8, 0.9]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                ],
                dtype=torch.uint8,
            ),
            "observation.images.image_is_pad": torch.tensor(
                [False, True], dtype=torch.bool
            ),
            "observation.state": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "observation.state_is_pad": torch.tensor([False], dtype=torch.bool),
            "action": torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32),
            "action_is_pad": torch.tensor([False, True], dtype=torch.bool),
            "task": "rotate",
            "episode_index": 3,
            "frame_index": 17,
        }


class _FakeFloatImageLeRobotDataset(_FakeLeRobotDataset):
    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        item["observation.images.image"] = (
            item["observation.images.image"].to(torch.float32) / 255.0
        )
        return item


class _FakeCustomActionKeyLeRobotDataset(_FakeLeRobotDataset):
    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        item["actions"] = item.pop("action")
        item["actions_is_pad"] = item.pop("action_is_pad")
        return item


def test_single_source_uses_lerobot_default_video_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "common.lerobot_v3_source.load_lerobot_meta",
        lambda repo_id, root, revision: make_test_meta(repo_id=repo_id),
    )
    monkeypatch.setattr("common.lerobot_v3_source.LeRobotDataset", _FakeLeRobotDataset)

    source = LeRobotSingleSource(
        repo_id="test/repo",
        root="/tmp/lerobot",
        revision="main",
        weight=1.0,
        camera_map={"primary": "observation.images.image"},
        state_key="observation.state",
        action_key="action",
    )
    source.compile(
        make_test_request(image_requests={"primary": (0, 1)}, action_deltas=(0, 1)),
        train_episode_indices={0, 1},
        val_episode_indices={0, 1},
    )
    source.prepare()

    source.get_sample(0)

    assert _FakeLeRobotDataset.init_kwargs is not None
    assert _FakeLeRobotDataset.last_instance is not None
    assert "video_backend" not in _FakeLeRobotDataset.init_kwargs
    assert set(_FakeLeRobotDataset.last_instance.meta.info["features"]) == {
        "observation.images.image",
        "observation.state",
        "action",
        "actions",
    }


def test_single_source_get_sample_shapes_images_masks_and_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "common.lerobot_v3_source.load_lerobot_meta",
        lambda repo_id, root, revision: make_test_meta(
            repo_id=repo_id,
            episodes=[
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 8}
            ],
        ),
    )
    monkeypatch.setattr("common.lerobot_v3_source.LeRobotDataset", _FakeLeRobotDataset)

    source = LeRobotSingleSource(
        repo_id="test/repo",
        root=None,
        revision=None,
        weight=1.0,
        camera_map={"primary": "observation.images.image"},
        state_key="observation.state",
        action_key="action",
    )
    source.compile(
        make_test_request(
            image_requests={"primary": (0, 1)},
            state_deltas=(0,),
            action_deltas=(0, 1),
            image_size=(4, 4),
        ),
        train_episode_indices={0},
        val_episode_indices={0},
    )
    source.prepare()

    sample = source.get_sample(0)

    assert sample.image_streams is not None
    assert sample.image_padding_masks is not None
    assert tuple(sample.image_streams["primary"].shape) == (2, 3, 4, 4)
    assert sample.image_streams["primary"].dtype == torch.uint8
    assert torch.equal(
        sample.image_padding_masks["primary"],
        torch.tensor([True, False], dtype=torch.bool),
    )
    assert tuple(sample.state.shape) == (1, 3)
    assert torch.equal(sample.state_is_pad, torch.tensor([False], dtype=torch.bool))
    assert torch.equal(
        sample.action_is_pad, torch.tensor([False, True], dtype=torch.bool)
    )
    assert sample.task_text == "rotate"
    assert sample.meta == {
        "dataset_name": "test/repo",
        "dataset_short": "repo",
        "episode_id": 3,
        "frame_idx": 17,
    }


def test_single_source_get_sample_accepts_float_image_streams(monkeypatch) -> None:
    monkeypatch.setattr(
        "common.lerobot_v3_source.load_lerobot_meta",
        lambda repo_id, root, revision: make_test_meta(
            repo_id=repo_id,
            episodes=[
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 8}
            ],
        ),
    )
    monkeypatch.setattr(
        "common.lerobot_v3_source.LeRobotDataset", _FakeFloatImageLeRobotDataset
    )

    source = LeRobotSingleSource(
        repo_id="test/repo",
        root=None,
        revision=None,
        weight=1.0,
        camera_map={"primary": "observation.images.image"},
        state_key="observation.state",
        action_key="action",
    )
    source.compile(
        make_test_request(
            image_requests={"primary": (0, 1)},
            image_size=(4, 4),
        ),
        train_episode_indices={0},
        val_episode_indices={0},
    )
    source.prepare()

    sample = source.get_sample(0)
    assert sample.image_streams is not None
    assert sample.image_streams["primary"].dtype == torch.uint8


def test_single_source_uses_custom_action_key_pad_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        "common.lerobot_v3_source.load_lerobot_meta",
        lambda repo_id, root, revision: make_test_meta(
            repo_id=repo_id,
            episodes=[
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 8}
            ],
        ),
    )
    monkeypatch.setattr(
        "common.lerobot_v3_source.LeRobotDataset", _FakeCustomActionKeyLeRobotDataset
    )

    source = LeRobotSingleSource(
        repo_id="test/repo",
        root=None,
        revision=None,
        weight=1.0,
        camera_map={"primary": "observation.images.image"},
        state_key="observation.state",
        action_key="actions",
    )
    source.compile(
        make_test_request(
            image_requests={"primary": (0, 1)},
            action_deltas=(0, 1),
        ),
        train_episode_indices={0},
        val_episode_indices={0},
    )
    source.prepare()

    sample = source.get_sample(0)

    assert torch.equal(
        sample.action_is_pad, torch.tensor([False, True], dtype=torch.bool)
    )


def test_single_source_dataset_short_uses_repo_suffix_without_alias_mapping(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "common.lerobot_v3_source.load_lerobot_meta",
        lambda repo_id, root, revision: make_test_meta(
            repo_id=repo_id,
            episodes=[
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 8}
            ],
        ),
    )
    monkeypatch.setattr("common.lerobot_v3_source.LeRobotDataset", _FakeLeRobotDataset)

    source = LeRobotSingleSource(
        repo_id="FedorX8/bridge_v2_lerobot",
        root=None,
        revision=None,
        weight=1.0,
        camera_map={"primary": "observation.images.image"},
        state_key="observation.state",
        action_key="action",
    )
    source.compile(
        make_test_request(
            image_requests={"primary": (0, 1)},
            state_deltas=(0,),
            action_deltas=(0, 1),
            image_size=(4, 4),
        ),
        train_episode_indices={0},
        val_episode_indices={0},
    )
    source.prepare()

    sample = source.get_sample(0)
    assert sample.meta is not None
    assert sample.meta["dataset_short"] == "bridge_v2"
