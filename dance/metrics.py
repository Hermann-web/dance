# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
import torchmetrics
from neuraltrain.metrics.base import BaseMetric
from neuraltrain.utils import convert_to_pydantic


class F1Event(torchmetrics.Metric):
    """F1 over event spans (start, end, class) at a fixed IoU threshold.

    A prediction is counted as a true positive only if it overlaps a
    ground-truth event by more than `iou_threshold` AND predicts the
    same class. Predictions are matched greedily to targets in order
    of descending confidence; once a target is matched it cannot be
    reused. Leftover predictions are false positives, leftover targets
    are false negatives.

    See section 4.2 of https://arxiv.org/abs/2605.10688 for the
    definition.

    Update inputs are lists of windows, each being a list of
    `(start, end, class_id, [confidence])` tuples. Predictions need a
    confidence in slot 3 (used for the greedy ranking); targets do not.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        for state in ("tp", "fp", "fn"):
            self.add_state(state, default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _iou(a, b):
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    def update(self, predicted_events, target_events):
        if target_events is None:
            return

        if predicted_events and isinstance(predicted_events[0], list):
            for pe, te in zip(predicted_events, target_events):
                self.update(pe, te)
            return

        if not predicted_events:
            self.fn += len(target_events)
            return
        if not target_events:
            self.fp += len(predicted_events)
            return

        preds = sorted(predicted_events, key=lambda x: x[3], reverse=True)
        matched_targets: set[int] = set()
        matched_preds = 0

        for p_start, p_end, p_cls, *_ in preds:
            best_iou, best_idx, best_cls = 0.0, -1, None
            for t_idx, (t_start, t_end, t_cls, *_) in enumerate(target_events):
                if t_idx in matched_targets:
                    continue
                iou = self._iou((p_start, p_end), (t_start, t_end))
                if iou > best_iou:
                    best_iou, best_idx, best_cls = iou, t_idx, t_cls

            if best_idx != -1 and best_iou >= self.iou_threshold:
                matched_targets.add(best_idx)
                matched_preds += 1
                if p_cls == best_cls:
                    self.tp += 1
                else:
                    self.fp += 1
                    self.fn += 1

        self.fp += len(preds) - matched_preds
        self.fn += len(target_events) - len(matched_targets)

    def compute(self):
        if (self.tp + self.fp + self.fn) <= 0:
            return torch.tensor(float("nan"), device=self.tp.device)
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)


F1EventConfig = convert_to_pydantic(
    F1Event,
    "F1Event",
    parent_class=BaseMetric,
    exclude_from_build=["log_name"],
)
