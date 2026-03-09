from __future__ import annotations

import torch

from common.lerobot_v3_types import BatchedDatasetSample, Stage1Batch
from stage2.backends.interfaces import Stage2Batch


def dataset_batch_to_stage1_batch(
    batch: BatchedDatasetSample,
    *,
    camera_roles: tuple[str, ...] | None = None,
) -> Stage1Batch:
    if batch.image_streams is None:
        raise ValueError("Expected image_streams in BatchedDatasetSample")
    if camera_roles is None:
        image_streams = dict(batch.image_streams)
        image_padding_masks = (
            None
            if batch.image_padding_masks is None
            else dict(batch.image_padding_masks)
        )
    else:
        image_streams = {role: batch.image_streams[role] for role in camera_roles}
        image_padding_masks = (
            None
            if batch.image_padding_masks is None
            else {role: batch.image_padding_masks[role] for role in camera_roles}
        )
    return Stage1Batch(
        image_streams=image_streams,
        image_padding_masks=image_padding_masks,
        task_text=batch.task_text,
        subtask_text=batch.subtask_text,
        state=batch.state,
        state_is_pad=batch.state_is_pad,
        action=batch.action,
        action_is_pad=batch.action_is_pad,
        meta=batch.meta,
    )


def dataset_batch_to_stage2_batch(batch: BatchedDatasetSample) -> Stage2Batch:
    return Stage2Batch(
        image_streams=(
            None if batch.image_streams is None else dict(batch.image_streams)
        ),
        image_padding_masks=(
            None
            if batch.image_padding_masks is None
            else dict(batch.image_padding_masks)
        ),
        task_text=batch.task_text,
        subtask_text=batch.subtask_text,
        state=batch.state,
        target_actions=batch.action,
        action_is_pad=batch.action_is_pad,
        meta=batch.meta,
    )
