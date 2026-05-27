# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
from neuralset.dataloader import Batch

from dance.training.data import detr_collate_fn


def _segment_stub():
    class _S:
        start = 0.0
        duration = 1.0

    return _S()


def _single_batch(*, q: int = 4, t: int = 8) -> Batch:
    return Batch(
        data={
            "neuro": torch.zeros(1, 16, 64),
            "feature_start": torch.rand(1, q, t),
            "feature_end": torch.rand(1, q, t),
            "feature_class": torch.randint(0, 3, (1, q, t)).float(),
            "dense_target": torch.zeros(1, 1, t),
        },
        segments=[_segment_stub()],
    )


def test_collate_renames_event_keys_and_collapses_time_axis():
    """feature_{start,end,class} -> *_target with the trailing samples
    axis squeezed; dense_target is dense (1, T) and is left alone."""
    out = detr_collate_fn([_single_batch(), _single_batch()])
    assert isinstance(out, Batch)
    assert out.data["neuro"].shape == (2, 16, 64)
    assert "feature_start" not in out.data
    assert "start_target" in out.data
    assert out.data["start_target"].shape == (2, 4)
    assert out.data["class_target"].shape == (2, 4)
    assert out.data["dense_target"].shape == (2, 1, 8)
