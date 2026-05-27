# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Unit tests for F1Event (the paper's F1-event metric)."""

from __future__ import annotations

import math

from dance.metrics import F1Event


def test_perfect_match_yields_f1_one():
    metric = F1Event(iou_threshold=0.5)
    preds = [(0.0, 1.0, 1, 0.9), (2.0, 3.0, 2, 0.9)]
    gts = [(0.0, 1.0, 1), (2.0, 3.0, 2)]
    metric.update(preds, gts)
    assert math.isclose(metric.compute().item(), 1.0, abs_tol=1e-6)


def test_class_mismatch_counts_as_fp_and_fn():
    """An IoU-good prediction with the wrong class is NOT a TP; it costs
    both an FP (the wrong-class prediction) and an FN (the unmatched GT)."""
    metric = F1Event(iou_threshold=0.5)
    preds = [(0.0, 1.0, 1, 0.9)]
    gts = [(0.0, 1.0, 2)]
    metric.update(preds, gts)
    assert metric.tp.item() == 0
    assert metric.fp.item() == 1
    assert metric.fn.item() == 1


def test_batched_update_iterates_over_windows():
    """update(list_of_window_lists, list_of_window_lists) accumulates each
    window's TP/FP/FN independently before the final F1 is computed."""
    metric = F1Event(iou_threshold=0.5)
    preds_batch = [
        [(0.0, 1.0, 1, 0.9)],
        [(0.0, 1.0, 2, 0.9), (2.0, 3.0, 1, 0.9)],
    ]
    gts_batch = [
        [(0.0, 1.0, 1)],
        [(0.0, 1.0, 2)],  # missing the second predicted event
    ]
    metric.update(preds_batch, gts_batch)
    assert metric.tp.item() == 2
    assert metric.fp.item() == 1
    assert metric.fn.item() == 0
