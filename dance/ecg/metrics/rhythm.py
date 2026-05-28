from __future__ import annotations

import torch
import torchmetrics

from ...metrics import F1Event


class EcgRhythmEpisodeF1(F1Event):
    """Interval F1 for rhythm episodes (e.g. AF)."""


class _BoundaryDelay(torchmetrics.Metric):
    def __init__(self, *, boundary: str) -> None:
        super().__init__()
        self.boundary = boundary
        self.add_state("sum_delay", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predicted_events, target_events) -> None:
        if predicted_events and isinstance(predicted_events[0], list):
            for pe, te in zip(predicted_events, target_events):
                self.update(pe, te)
            return
        n = min(len(predicted_events), len(target_events))
        for i in range(n):
            p = predicted_events[i]
            t = target_events[i]
            idx = 0 if self.boundary == "start" else 1
            self.sum_delay += abs(float(p[idx]) - float(t[idx]))
            self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.tensor(float("nan"), device=self.sum_delay.device)
        return self.sum_delay / self.count


class EcgOnsetDelay(_BoundaryDelay):
    def __init__(self) -> None:
        super().__init__(boundary="start")


class EcgOffsetDelay(_BoundaryDelay):
    def __init__(self) -> None:
        super().__init__(boundary="end")


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
