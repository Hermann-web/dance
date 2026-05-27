# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def is_rank_zero() -> bool:
    """Return True on the global-rank-zero process (or when not using DDP)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


def events_to_mask(
    events,
    duration: float,
    frequency: float,
    num_classes: int,
) -> torch.Tensor:
    """Render `(start, end, class_id, ...)` tuples into a per-timestep
    `(num_classes, T)` multilabel mask. Background (row 0) is set
    wherever no other class fires. Consumed by `MultilabelF1Score`.
    """
    n = int(duration * frequency)
    mask = torch.zeros((num_classes, n), dtype=torch.long)
    for ev in events:
        s = max(0, int(ev[0] * frequency))
        e = min(n, int(ev[1] * frequency))
        cls = int(ev[2])
        if 0 < cls < num_classes:
            mask[cls, s:e] = 1
    active = mask[1:].sum(dim=0).clamp(max=1)
    mask[0] = 1 - active
    return mask


def make_masks(
    events_per_window,
    device: torch.device,
    duration: float,
    frequency: float,
    num_classes: int,
) -> torch.Tensor:
    """Batched events_to_mask: stack one mask per window."""
    return torch.stack(
        [events_to_mask(ev, duration, frequency, num_classes) for ev in events_per_window]
    ).to(device)


def detr_to_dense_probs(
    preds: dict[str, torch.Tensor],
    duration: float,
    frequency: float,
    n_classes: int,
) -> torch.Tensor:
    """Project the DETR query outputs into a dense (batch, T, n_classes)
    probability map.

    For every window, each query's class softmax is accumulated into
    the timesteps covered by its predicted interval, then renormalised
    so each timestep is a proper distribution over classes. This is
    what the consistency loss compares (in KL) against the dense head's
    own per-timestep softmax.
    """
    B, Q, _ = preds["class"].shape
    T = int(duration * frequency)
    device = preds["class"].device

    out = []
    for b in range(B):
        mask = torch.zeros(T, n_classes, device=device)
        for q in range(Q):
            probs = torch.softmax(preds["class"][b, q], dim=-1)
            s = preds["start"][b, q].item()
            e = preds["end"][b, q].item()
            start = int(max(0, s * T))
            end = int(min(T, e * T))
            if start < end:
                mask[start:end] += probs.unsqueeze(0)
        out.append(mask / (mask.sum(dim=-1, keepdim=True) + 1e-8))
    return torch.stack(out, dim=0)


def extract_events_from_detr_batch(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    window_length: float,
    *,
    to_numpy: bool = False,
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """Decode DETR predictions + zero-padded targets into per-window
    event lists.

    Returns (pred_events_per_window, gt_events_per_window) where each
    inner list contains (start_seconds, end_seconds, class_id,
    [confidence]) tuples. Predictions carry a confidence in slot 3
    (used by F1Event to rank them), ground-truth events do not.

    Consumed by the metric updater in BrainModule and by the
    PredictedEvents callback that dumps per-batch predictions to JSON.
    """

    def _maybe_cpu(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if to_numpy:
                x = x.numpy()
        return x

    def _to_intervals(start, end):
        s = float(max(0.0, start * window_length))
        e = float(min(float(window_length), end * window_length))
        return s, e

    logits_preds = _maybe_cpu(preds["class"])
    class_targets = _maybe_cpu(targets["class"])
    p_a, p_b = _maybe_cpu(preds["start"]), _maybe_cpu(preds["end"])
    t_a, t_b = _maybe_cpu(targets["start"]), _maybe_cpu(targets["end"])

    pred_per_window: list[list[tuple]] = []
    gt_per_window: list[list[tuple]] = []
    for i in range(logits_preds.shape[0]):
        gt = []
        for a, b, cls in zip(t_a[i], t_b[i], class_targets[i]):
            if int(cls) <= 0:
                continue
            s, e = _to_intervals(a, b)
            gt.append((s, e, int(cls)))
        gt_per_window.append(gt)

        probs = torch.softmax(torch.as_tensor(logits_preds[i]), dim=-1)
        scores, labels = probs.max(dim=-1)
        preds_window = []
        for q, label in enumerate(labels.tolist()):
            if label == 0:
                continue
            s, e = _to_intervals(p_a[i][q], p_b[i][q])
            preds_window.append((s, e, label, float(scores[q])))
        pred_per_window.append(preds_window)

    return pred_per_window, gt_per_window


def seizure_weighted_sampler(
    train_segments,
    *,
    alpha: float = 0.01,
    beta: float = 5.0,
) -> WeightedRandomSampler:
    """Sample training windows by seizure occupancy.

    Each window's sampling weight is
    `alpha + (1 - alpha) * min(1, beta * occ)`, where `occ` is the
    fraction of the window covered by seizure events. Used only by
    the TUSZ training loaders.
    """
    occupancies = np.array(
        [
            seg.events[seg.events["type"] == "Seizure"]["duration"].sum()
            / max(seg.duration, 1e-6)
            for seg in train_segments
        ]
    )
    weights = alpha + (1 - alpha) * np.minimum(1.0, beta * occupancies)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )
