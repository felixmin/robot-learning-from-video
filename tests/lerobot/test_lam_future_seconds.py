from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
LAM_SRC = Path("/mnt/data/workspace/code/lerobot_policy_lapa_lam/src")
for path in (LAM_SRC, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot_policy_lapa_lam.configuration_lam import LAMConfig


class _FakeMeta:
    def __init__(self, fps: float):
        self.fps = fps
        self.features = {"observation.images.image": {"dtype": "image", "shape": [256, 256, 3]}}


def test_lam_future_seconds_resolves_source_local_frame_offsets():
    cfg = LAMConfig(future_frames=1, future_seconds=0.5)

    assert cfg.get_observation_delta_indices_for_fps(3) == [0, 2]
    assert cfg.get_observation_delta_indices_for_fps(10) == [0, 5]
    assert cfg.get_observation_delta_indices_for_fps(20) == [0, 10]


def test_resolve_delta_timestamps_uses_future_seconds_per_source_fps():
    cfg = LAMConfig(future_frames=1, future_seconds=0.5)

    slow = resolve_delta_timestamps(cfg, _FakeMeta(fps=3))
    fast = resolve_delta_timestamps(cfg, _FakeMeta(fps=20))

    assert slow["observation.images.image"] == [0.0, 2 / 3]
    assert fast["observation.images.image"] == [0.0, 0.5]
