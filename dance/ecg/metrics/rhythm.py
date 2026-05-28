from __future__ import annotations

import torch
import torchmetrics

from ...metrics import F1Event


class EcgRhythmEpisodeF1(F1Event):
    """Interval F1 for rhythm episodes (e.g. AF)."""


class _BoundaryDelay(torchmetrics.Metric):
    def __init__(self, *, boundary: str, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.boundary = boundary
        self.iou_threshold = iou_threshold
        self.add_state("sum_delay", default=torch.tensor(0.0), dist_reduce_fx="sum")
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
        matched_targets: set[int] = set()
        preds = sorted(predicted_events, key=lambda x: x[3], reverse=True)
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
                idx = 0 if self.boundary == "start" else 1
                pred_boundary = (p_start, p_end)[idx]
                tgt_boundary = best_target[idx]
                self.sum_delay += abs(float(pred_boundary) - float(tgt_boundary))
                self.count += 1
                matched_targets.add(best_idx)

    def compute(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.tensor(float("nan"), device=self.sum_delay.device)
        return self.sum_delay / self.count


class EcgOnsetDelay(_BoundaryDelay):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__(boundary="start", iou_threshold=iou_threshold)


class EcgOffsetDelay(_BoundaryDelay):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__(boundary="end", iou_threshold=iou_threshold)


class EcgBurdenError(torchmetrics.Metric):
    """Absolute burden error on normalized timeline [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predicted_events, target_events) -> None:
        if predicted_events and isinstance(predicted_events[0], list):
            for pe, te in zip(predicted_events, target_events):
                self.update(pe, te)
            return
        p_burden = sum(max(0.0, float(e[1]) - float(e[0])) for e in predicted_events)
        t_burden = sum(max(0.0, float(e[1]) - float(e[0])) for e in target_events)
        self.sum_error += abs(p_burden - t_burden)
        self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.tensor(float("nan"), device=self.sum_error.device)
        return self.sum_error / self.count
