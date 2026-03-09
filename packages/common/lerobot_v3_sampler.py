from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from common.lerobot_v3_source import CompiledSourceIndex
from common.lerobot_v3_stats import normalize_weights
from common.lerobot_v3_types import SampleToken


@dataclass
class _SourceCycleState:
    order: np.ndarray
    pointer: int
    episode_row_by_id: dict[int, int]


class WeightedLeRobotTokenSampler(torch.utils.data.Sampler[SampleToken]):
    def __init__(
        self,
        *,
        compiled_sources: list[CompiledSourceIndex],
        source_weights: np.ndarray,
        num_samples: int,
        seed: int,
        epoch: int,
        resample_each_epoch: bool,
    ) -> None:
        self.compiled_sources = compiled_sources
        self.source_weights = normalize_weights(
            np.asarray(source_weights, dtype=np.float64)
        )
        if len(compiled_sources) != int(self.source_weights.shape[0]):
            raise ValueError("compiled_sources and source_weights length mismatch")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.resample_each_epoch = bool(resample_each_epoch)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        seed = self.seed + self.epoch if self.resample_each_epoch else self.seed
        return np.random.default_rng(seed)

    def _build_source_states(self, rng: np.random.Generator) -> list[_SourceCycleState]:
        states: list[_SourceCycleState] = []
        for source in self.compiled_sources:
            sampleable = np.asarray(source.sampleable_episode_ids, dtype=np.int32)
            if sampleable.size == 0:
                raise ValueError(
                    f"Source {source.repo_id!r} has no sampleable episodes"
                )
            order = np.asarray(
                sampleable[rng.permutation(sampleable.size)], dtype=np.int32
            )
            episode_row_by_id = {
                int(ep_id): idx
                for idx, ep_id in enumerate(source.episodes.episode_index.tolist())
            }
            states.append(
                _SourceCycleState(
                    order=order,
                    pointer=0,
                    episode_row_by_id=episode_row_by_id,
                )
            )
        return states

    def _next_episode_id(
        self,
        *,
        source_id: int,
        state: _SourceCycleState,
        rng: np.random.Generator,
    ) -> int:
        if state.pointer >= int(state.order.shape[0]):
            state.order = np.asarray(
                state.order[rng.permutation(state.order.shape[0])], dtype=np.int32
            )
            state.pointer = 0
        episode_id = int(state.order[state.pointer])
        state.pointer += 1
        return episode_id

    def _sample_token_from_source(
        self,
        *,
        source_id: int,
        state: _SourceCycleState,
        rng: np.random.Generator,
    ) -> SampleToken:
        source = self.compiled_sources[source_id]
        episode_id = self._next_episode_id(source_id=source_id, state=state, rng=rng)
        row_idx = state.episode_row_by_id[episode_id]
        start = int(source.episodes.valid_anchor_start[row_idx])
        end = int(source.episodes.valid_anchor_end[row_idx])
        if end <= start:
            raise ValueError(
                f"Episode {episode_id} of source {source.repo_id!r} has no valid anchors: [{start}, {end})"
            )
        anchor = int(rng.integers(start, end))
        return SampleToken(
            source_id=source_id, episode_id=episode_id, anchor_abs_index=anchor
        )

    def _build_epoch_plan(self, total_samples: int) -> list[SampleToken]:
        rng = self._rng()
        source_states = self._build_source_states(rng)
        plan: list[SampleToken] = []
        for _ in range(int(total_samples)):
            source_id = int(
                rng.choice(len(self.compiled_sources), p=self.source_weights)
            )
            token = self._sample_token_from_source(
                source_id=source_id,
                state=source_states[source_id],
                rng=rng,
            )
            plan.append(token)
        return plan

    def __iter__(self) -> Iterator[SampleToken]:
        return iter(self._build_epoch_plan(self.num_samples))


class DistributedWeightedLeRobotTokenSampler(WeightedLeRobotTokenSampler):
    def __init__(
        self,
        *,
        compiled_sources: list[CompiledSourceIndex],
        source_weights: np.ndarray,
        global_num_samples: int,
        rank: int,
        world_size: int,
        seed: int,
        epoch: int,
        resample_each_epoch: bool,
    ) -> None:
        if int(global_num_samples) % int(world_size) != 0:
            raise ValueError("global_num_samples must be divisible by world_size")
        self.global_num_samples = int(global_num_samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        super().__init__(
            compiled_sources=compiled_sources,
            source_weights=source_weights,
            num_samples=self.global_num_samples // self.world_size,
            seed=seed,
            epoch=epoch,
            resample_each_epoch=resample_each_epoch,
        )

    def __iter__(self) -> Iterator[SampleToken]:
        global_plan = self._build_epoch_plan(self.global_num_samples)
        return iter(global_plan[self.rank :: self.world_size])
