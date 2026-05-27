# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn

from dance.losses import ConsistencyLoss, DetrLoss, IoULoss
from dance.matcher import HungarianMatcher


def test_detr_loss_returns_scalar_total_and_details():
    """DetrLoss wires the matcher + class CE + IoU loss into one scalar."""
    matcher = HungarianMatcher(weight_class=1.0, weight_iou=1.0)
    loss = DetrLoss(
        matcher=matcher,
        class_loss=nn.CrossEntropyLoss(),
        iou_loss=IoULoss(mode="start_end").build(),
        weight_class=1.0,
        weight_iou=5.0,
    )
    preds = {
        "class": torch.tensor([[[0.0, 5.0], [5.0, 0.0]]]),
        "start": torch.tensor([[0.1, 0.5]]),
        "end": torch.tensor([[0.2, 0.6]]),
    }
    targets = {
        "class": torch.tensor([[1, 0]]),
        "start": torch.tensor([[0.1, 0.0]]),
        "end": torch.tensor([[0.2, 0.0]]),
    }
    total, details = loss(preds, targets)
    assert total.ndim == 0
    assert {"class_loss", "iou_loss"} <= details.keys()


def test_consistency_loss_is_zero_when_dense_matches_detr_induced():
    """If the dense softmax and the DETR-induced dense distribution are
    both uniform, their KL is 0. Validates the sparse->dense bridge.
    """
    loss = ConsistencyLoss(
        nn.KLDivLoss(reduction="none"),
        n_classes=3,
        duration=1.0,
        frequency=8.0,
        weight=1.0,
    )
    preds = {
        "dense": torch.zeros(1, 8, 3),
        "class": torch.zeros(1, 4, 3),
        "start": torch.zeros(1, 4),
        "end": torch.ones(1, 4),
    }
    assert loss(preds).abs().item() < 1e-4
