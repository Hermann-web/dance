from __future__ import annotations

import torch
import torchmetrics

from ...metrics import F1Event


class EcgWaveDelineationF1(F1Event):
    """F1 over ECG wave intervals (P/QRS/T) with IoU matching."""


class _MatchedBoundaryMetric(torchmetrics.Metric):
    def __init__(self, *, boundary: str, iou_threshold: float = 0.5) -> None:
        super().__init__()
        if boundary not in {"start", "end"}:
            raise ValueError("boundary must be 'start' or 'end'")
        self.boundary = boundary
        self.iou_threshold = iou_threshold
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    def update(self, predicted_events, target_events) -> None:
        if predicted_events and isinstance(predicted_events[0], list):
            for pe, te in zip(predicted_events, target_events):
                self.update(pe, te)
            return
        if not predicted_events or not target_events:
            return
        preds = sorted(predicted_events, key=lambda x: x[3], reverse=True)
        matched_targets: set[int] = set()
        for p_start, p_end, p_cls, *_ in preds:
            best_iou, best_idx = 0.0, -1
            best_target = None
            for t_idx, (t_start, t_end, t_cls, *_) in enumerate(target_events):
                if t_idx in matched_targets or p_cls != t_cls:
                    continue
                iou = self._iou((p_start, p_end), (t_start, t_end))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = t_idx
                    best_target = (t_start, t_end)
            if best_idx != -1 and best_iou >= self.iou_threshold and best_target is not None:
                matched_targets.add(best_idx)
                pred_boundary = p_start if self.boundary == "start" else p_end
                tgt_boundary = best_target[0] if self.boundary == "start" else best_target[1]
                self.sum_abs_error += abs(pred_boundary - tgt_boundary)
                self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.tensor(float("nan"), device=self.sum_abs_error.device)
        return self.sum_abs_error / self.count


class EcgOnsetMAE(_MatchedBoundaryMetric):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__(boundary="start", iou_threshold=iou_threshold)


class EcgOffsetMAE(_MatchedBoundaryMetric):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__(boundary="end", iou_threshold=iou_threshold)


class EcgToleranceF1(torchmetrics.Metric):
    """F1 where class-equal onset/offset errors must both be <= tolerance."""

    def __init__(self, tolerance: float = 0.02) -> None:
        super().__init__()
        self.tolerance = tolerance
        self._eps = 1e-8
        for state in ("tp", "fp", "fn"):
            self.add_state(state, default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predicted_events, target_events) -> None:
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
        matched_targets: set[int] = set()
        matched_preds = 0
        preds = sorted(predicted_events, key=lambda x: x[3], reverse=True)
        for p_start, p_end, p_cls, *_ in preds:
            best_idx = -1
            for t_idx, (t_start, t_end, t_cls, *_) in enumerate(target_events):
                if t_idx in matched_targets or p_cls != t_cls:
                    continue
                if (
                    abs(p_start - t_start) <= self.tolerance + self._eps
                    and abs(p_end - t_end) <= self.tolerance + self._eps
                ):
                    best_idx = t_idx
                    break
            if best_idx != -1:
                matched_targets.add(best_idx)
                matched_preds += 1
                self.tp += 1
        self.fp += len(preds) - matched_preds
        self.fn += len(target_events) - len(matched_targets)

    def compute(self) -> torch.Tensor:
        if (self.tp + self.fp + self.fn) <= 0:
            return torch.tensor(float("nan"), device=self.tp.device)
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)


def as_event_lists(batch: dict[str, torch.Tensor], scores: torch.Tensor | None = None):
    start, end, cls = batch["start"], batch["end"], batch["class"]
    out = []
    for b in range(start.shape[0]):
        ev = []
        for i in range(start.shape[1]):
            c = int(cls[b, i])
            if c == 0:
                continue
            tup = (float(start[b, i]), float(end[b, i]), c)
            if scores is not None:
                tup = (*tup, float(scores[b, i]))
            ev.append(tup)
        out.append(ev)
    return out
