from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .. import utils
from .events.schema import WAVE_CLASS_TO_ID
from .metrics import (
    EcgBurdenError,
    EcgOffsetDelay,
    EcgOffsetMAE,
    EcgOnsetDelay,
    EcgOnsetMAE,
    EcgRhythmEpisodeF1,
    EcgToleranceF1,
    EcgWaveDelineationF1,
)


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _event_iou(pred: tuple, target: tuple) -> float:
    inter = max(0.0, min(float(pred[1]), float(target[1])) - max(float(pred[0]), float(target[0])))
    union = max(float(pred[1]), float(target[1])) - min(float(pred[0]), float(target[0]))
    return inter / union if union > 0 else 0.0


def mean_matched_iou(
    predicted_events,
    target_events,
    *,
    iou_threshold: float = 0.0,
) -> float:
    if predicted_events and isinstance(predicted_events[0], list):
        values = [
            mean_matched_iou(preds, targets, iou_threshold=iou_threshold)
            for preds, targets in zip(predicted_events, target_events)
        ]
        values = [value for value in values if not math.isnan(value)]
        return float(np.mean(values)) if values else float("nan")
    if not predicted_events or not target_events:
        return float("nan")

    preds = sorted(predicted_events, key=lambda x: x[3], reverse=True)
    matched_targets: set[int] = set()
    ious: list[float] = []
    for pred in preds:
        best_iou, best_idx = 0.0, -1
        for target_idx, target in enumerate(target_events):
            if target_idx in matched_targets or int(pred[2]) != int(target[2]):
                continue
            iou = _event_iou(pred, target)
            if iou > best_iou:
                best_iou = iou
                best_idx = target_idx
        if best_idx != -1 and best_iou >= iou_threshold:
            matched_targets.add(best_idx)
            ious.append(best_iou)
    return float(np.mean(ious)) if ious else float("nan")


def _events_to_sample_labels(
    events: Sequence[tuple],
    *,
    n_samples: int,
    duration: float,
) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=np.int64)
    if n_samples <= 0 or duration <= 0:
        return labels
    for event in events:
        start_s, end_s, class_id = float(event[0]), float(event[1]), int(event[2])
        start = int(np.floor((start_s / duration) * n_samples))
        end = int(np.ceil((end_s / duration) * n_samples))
        start = max(0, min(n_samples, start))
        end = max(0, min(n_samples, end))
        if class_id > 0 and end > start:
            labels[start:end] = class_id
    return labels


def samplewise_multiclass_metrics(
    predicted_events_per_window: Sequence[Sequence[tuple]],
    target_events_per_window: Sequence[Sequence[tuple]],
    *,
    n_samples_per_window: Sequence[int],
    duration: float,
    positive_class_ids: Sequence[int],
) -> dict[str, float]:
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_specificity: list[float] = []
    per_class_f1: list[float] = []
    total_correct = 0
    total_samples = 0

    for class_id in positive_class_ids:
        tp = fp = tn = fn = 0
        for pred_events, target_events, n_samples in zip(
            predicted_events_per_window,
            target_events_per_window,
            n_samples_per_window,
        ):
            pred = _events_to_sample_labels(pred_events, n_samples=n_samples, duration=duration)
            target = _events_to_sample_labels(target_events, n_samples=n_samples, duration=duration)
            pred_pos = pred == class_id
            target_pos = target == class_id
            tp += int(np.logical_and(pred_pos, target_pos).sum())
            fp += int(np.logical_and(pred_pos, ~target_pos).sum())
            tn += int(np.logical_and(~pred_pos, ~target_pos).sum())
            fn += int(np.logical_and(~pred_pos, target_pos).sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        specificity = _safe_ratio(tn, tn + fp)
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) <= 0:
            f1 = float("nan")
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_specificity.append(specificity)
        per_class_f1.append(f1)

    for pred_events, target_events, n_samples in zip(
        predicted_events_per_window,
        target_events_per_window,
        n_samples_per_window,
    ):
        pred = _events_to_sample_labels(pred_events, n_samples=n_samples, duration=duration)
        target = _events_to_sample_labels(target_events, n_samples=n_samples, duration=duration)
        total_correct += int((pred == target).sum())
        total_samples += int(n_samples)

    return {
        "sample_accuracy": _safe_ratio(total_correct, total_samples),
        "sample_macro_precision": float(np.nanmean(per_class_precision)),
        "sample_macro_sensitivity": float(np.nanmean(per_class_recall)),
        "sample_macro_specificity": float(np.nanmean(per_class_specificity)),
        "sample_macro_f1": float(np.nanmean(per_class_f1)),
    }


def evaluate_wave_events(
    predicted_events_per_window: Sequence[Sequence[tuple]],
    target_events_per_window: Sequence[Sequence[tuple]],
    *,
    duration: float,
    n_samples_per_window: Sequence[int],
    iou_threshold: float = 0.5,
    tolerance: float = 0.02,
) -> dict[str, float]:
    interval_f1 = EcgWaveDelineationF1(iou_threshold=iou_threshold)
    onset_mae = EcgOnsetMAE(iou_threshold=iou_threshold)
    offset_mae = EcgOffsetMAE(iou_threshold=iou_threshold)
    tolerance_f1 = EcgToleranceF1(tolerance=tolerance)
    for metric in (interval_f1, onset_mae, offset_mae, tolerance_f1):
        metric.update(predicted_events_per_window, target_events_per_window)

    metrics = {
        "event_f1": _to_float(interval_f1.compute()),
        "onset_mae": _to_float(onset_mae.compute()),
        "offset_mae": _to_float(offset_mae.compute()),
        "tolerance_f1": _to_float(tolerance_f1.compute()),
    }
    metrics.update(
        samplewise_multiclass_metrics(
            predicted_events_per_window,
            target_events_per_window,
            n_samples_per_window=n_samples_per_window,
            duration=duration,
            positive_class_ids=[
                class_id for class_name, class_id in WAVE_CLASS_TO_ID.items() if class_name != "bg"
            ],
        )
    )
    return metrics


def evaluate_rhythm_events(
    predicted_events_per_window: Sequence[Sequence[tuple]],
    target_events_per_window: Sequence[Sequence[tuple]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    episode_f1 = EcgRhythmEpisodeF1(iou_threshold=iou_threshold)
    onset_delay = EcgOnsetDelay(iou_threshold=iou_threshold)
    offset_delay = EcgOffsetDelay(iou_threshold=iou_threshold)
    burden_error = EcgBurdenError()
    for metric in (episode_f1, onset_delay, offset_delay, burden_error):
        metric.update(predicted_events_per_window, target_events_per_window)
    return {
        "episode_f1": _to_float(episode_f1.compute()),
        "onset_delay": _to_float(onset_delay.compute()),
        "offset_delay": _to_float(offset_delay.compute()),
        "burden_error": _to_float(burden_error.compute()),
        "mean_matched_iou": mean_matched_iou(
            predicted_events_per_window,
            target_events_per_window,
            iou_threshold=iou_threshold,
        ),
    }


def _resolve_n_samples_per_window(fs: float | torch.Tensor | Sequence[float], batch_size: int, duration: float):
    if isinstance(fs, torch.Tensor):
        values = fs.detach().cpu().tolist()
    elif isinstance(fs, Sequence) and not isinstance(fs, (str, bytes)):
        values = list(fs)
    else:
        values = [float(fs)] * batch_size
    return [max(1, int(round(float(value) * duration))) for value in values]


def evaluate_wave_batch(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    duration: float,
    fs: float | torch.Tensor | Sequence[float],
    iou_threshold: float = 0.5,
    tolerance: float = 0.02,
) -> dict[str, float]:
    predicted_events, target_events = utils.extract_events_from_detr_batch(
        preds,
        targets,
        duration,
    )
    n_samples_per_window = _resolve_n_samples_per_window(fs, len(predicted_events), duration)
    return evaluate_wave_events(
        predicted_events,
        target_events,
        duration=duration,
        n_samples_per_window=n_samples_per_window,
        iou_threshold=iou_threshold,
        tolerance=tolerance,
    )


def evaluate_rhythm_batch(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    duration: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    predicted_events, target_events = utils.extract_events_from_detr_batch(
        preds,
        targets,
        duration,
    )
    return evaluate_rhythm_events(
        predicted_events,
        target_events,
        iou_threshold=iou_threshold,
    )


__all__ = [
    "evaluate_rhythm_batch",
    "evaluate_rhythm_events",
    "evaluate_wave_batch",
    "evaluate_wave_events",
    "mean_matched_iou",
    "samplewise_multiclass_metrics",
]
