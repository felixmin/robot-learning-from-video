from __future__ import annotations

import pytest

from common.lerobot_v3_source import resolve_request_to_delta_timestamps

from tests.helpers.lerobot_v3_fixtures import make_test_request


def test_resolve_request_to_delta_timestamps_maps_camera_roles() -> None:
    request = make_test_request(
        image_requests={"primary": (0, 5), "wrist": (0, 10)},
        state_deltas=(0,),
        action_deltas=(0, 1, 2),
    )

    out = resolve_request_to_delta_timestamps(
        request=request,
        fps=10,
        camera_role_to_key={
            "primary": "observation.images.image",
            "wrist": "observation.images.image_wrist",
        },
        state_key="observation.state",
        action_key="action",
    )

    assert out == {
        "observation.images.image": [0.0, 0.5],
        "observation.images.image_wrist": [0.0, 1.0],
        "observation.state": [0.0],
        "action": [0.0, 0.1, 0.2],
    }


def test_resolve_request_to_delta_timestamps_converts_steps_using_fps() -> None:
    request = make_test_request(image_requests={"primary": (-2, 0, 5)})

    out = resolve_request_to_delta_timestamps(
        request=request,
        fps=20,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key=None,
        action_key=None,
    )

    assert out["observation.images.rgb"] == [-0.1, 0.0, 0.25]


def test_resolve_request_to_delta_timestamps_omits_unrequested_modalities() -> None:
    request = make_test_request(
        image_requests={"primary": (0,)}, state_deltas=None, action_deltas=None
    )

    out = resolve_request_to_delta_timestamps(
        request=request,
        fps=5,
        camera_role_to_key={"primary": "observation.images.rgb"},
        state_key="observation.state",
        action_key="action",
    )

    assert out == {"observation.images.rgb": [0.0]}


def test_resolve_request_to_delta_timestamps_rejects_missing_required_camera_role() -> (
    None
):
    request = make_test_request(image_requests={"primary": (0,), "wrist": (0,)})

    with pytest.raises(KeyError, match="wrist"):
        resolve_request_to_delta_timestamps(
            request=request,
            fps=10,
            camera_role_to_key={"primary": "observation.images.rgb"},
            state_key=None,
            action_key=None,
        )
