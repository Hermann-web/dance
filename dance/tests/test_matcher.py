# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Unit tests for HungarianMatcher (the heart of the DETR loss path)."""

from __future__ import annotations

import torch

from dance.matcher import HungarianMatcher


def test_known_optimal_assignment():
    """Two queries, two targets; the unique optimal assignment is identity."""
    matcher = HungarianMatcher(weight_class=1.0, weight_iou=1.0, window_length=1.0)
    outputs = {
        "class": torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0],  # query 0 strongly predicts class 1
                    [0.0, 0.0, 5.0],  # query 1 strongly predicts class 2
                ]
            ]
        ),
        "start": torch.tensor([[0.10, 0.50]]),
        "end": torch.tensor([[0.20, 0.60]]),
    }
    targets = {
        "class": torch.tensor([[1, 2]]),
        "start": torch.tensor([[0.10, 0.50]]),
        "end": torch.tensor([[0.20, 0.60]]),
    }
    _, matched_targets, matches = matcher(outputs, targets)
    q_idx = matches[0]["q_idx"].tolist()
    t_idx = matches[0]["t_idx"].tolist()
    assert sorted(q_idx) == [0, 1]
    assert dict(zip(q_idx, t_idx)) == {0: 0, 1: 1}
    assert matched_targets["class"][0].tolist() == [1, 2]


def test_no_targets_returns_empty_match():
    """All-background targets must produce an empty match list."""
    matcher = HungarianMatcher(weight_class=1.0, weight_iou=1.0)
    outputs = {
        "class": torch.zeros(1, 3, 2),
        "start": torch.zeros(1, 3),
        "end": torch.zeros(1, 3),
    }
    targets = {
        "class": torch.zeros(1, 3, dtype=torch.long),
        "start": torch.zeros(1, 3),
        "end": torch.zeros(1, 3),
    }
    _, matched_targets, matches = matcher(outputs, targets)
    assert matches[0]["q_idx"].numel() == 0
    assert matched_targets["class"].sum().item() == 0
