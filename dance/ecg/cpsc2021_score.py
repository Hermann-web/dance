from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RHYTHM_SCORE_MATRIX = np.array([[1.0, -1.0, -0.5], [-2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])


def _class_from_comments(comments: list[str]) -> int:
    joined = " ".join(comments).strip().lower()
    if "non atrial fibrillation" in joined:
        return 0
    if "persistent atrial fibrillation" in joined:
        return 1
    if "paroxysmal atrial fibrillation" in joined:
        return 2
    raise ValueError(f"Unsupported CPSC2021 rhythm comments: {comments!r}")


def classify_prediction(endpoints: np.ndarray, signal_length: int) -> int:
    if len(endpoints) == 0:
        return 0
    if len(endpoints) == 1 and int(endpoints[0][1]) - int(endpoints[0][0]) == signal_length - 1:
        return 1
    return 2


def _safe_beat(beat_locations: np.ndarray, idx: int) -> int:
    idx = max(0, min(len(beat_locations) - 1, idx))
    return int(beat_locations[idx])


@dataclass(slots=True)
class Cpsc2021ReferenceInfo:
    sample_path: str
    fs: float
    signal_length: int
    beat_locations: np.ndarray
    af_start_indices: np.ndarray
    af_end_indices: np.ndarray
    class_true: int
    onset_score_range: np.ndarray | None
    offset_score_range: np.ndarray | None

    @property
    def endpoints_true(self) -> np.ndarray:
        return np.dstack((self.af_start_indices, self.af_end_indices))[0, :, :]


def load_reference_info(record_path: str | Path) -> Cpsc2021ReferenceInfo:
    import wfdb

    sample_path = str(record_path)
    record = wfdb.rdrecord(sample_path)
    ann_ref = wfdb.rdann(sample_path, "atr")

    beat_locations = np.asarray(ann_ref.sample, dtype=np.int64)
    ann_note = np.asarray(ann_ref.aux_note)
    af_start_indices = np.where((ann_note == "(AFIB") | (ann_note == "(AFL"))[0]
    af_end_indices = np.where(ann_note == "(N")[0]
    class_true = _class_from_comments(list(record.comments))
    if class_true in {1, 2}:
        onset_range, offset_range = generate_endpoint_score_ranges(
            beat_locations=beat_locations,
            af_start_indices=af_start_indices,
            af_end_indices=af_end_indices,
            class_true=class_true,
            signal_length=int(record.sig_len),
        )
    else:
        onset_range = None
        offset_range = None
    return Cpsc2021ReferenceInfo(
        sample_path=sample_path,
        fs=float(record.fs),
        signal_length=int(record.sig_len),
        beat_locations=beat_locations,
        af_start_indices=af_start_indices,
        af_end_indices=af_end_indices,
        class_true=class_true,
        onset_score_range=onset_range,
        offset_score_range=offset_range,
    )


def generate_endpoint_score_ranges(
    *,
    beat_locations: np.ndarray,
    af_start_indices: np.ndarray,
    af_end_indices: np.ndarray,
    class_true: int,
    signal_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    onset_range = np.zeros((signal_length,), dtype=np.float64)
    offset_range = np.zeros((signal_length,), dtype=np.float64)

    for af_start in af_start_indices:
        af_start = int(af_start)
        if class_true == 2:
            left2 = max(af_start - 2, 0)
            left1 = max(af_start - 1, 0)
            right2 = min(af_start + 2, len(beat_locations) - 1)
            right3 = min(af_start + 3, len(beat_locations) - 1)
            if left1 == 0:
                onset_range[: _safe_beat(beat_locations, right2)] += 1.0
            elif left2 == 0:
                onset_range[_safe_beat(beat_locations, left1) : _safe_beat(beat_locations, right2)] += 1.0
                onset_range[: _safe_beat(beat_locations, left1)] += 0.5
            else:
                onset_range[_safe_beat(beat_locations, left1) : _safe_beat(beat_locations, right2)] += 1.0
                onset_range[_safe_beat(beat_locations, left2) : _safe_beat(beat_locations, left1)] += 0.5
            onset_range[_safe_beat(beat_locations, right2) : _safe_beat(beat_locations, right3)] += 0.5
        elif class_true == 1:
            right2 = min(af_start + 2, len(beat_locations) - 1)
            right3 = min(af_start + 3, len(beat_locations) - 1)
            onset_range[: _safe_beat(beat_locations, right2)] += 1.0
            onset_range[_safe_beat(beat_locations, right2) : _safe_beat(beat_locations, right3)] += 0.5

    for af_end in af_end_indices:
        af_end = int(af_end)
        if class_true == 2:
            left3 = max(af_end - 3, 0)
            left2 = max(af_end - 2, 0)
            right1 = min(af_end + 1, len(beat_locations) - 1)
            right2 = min(af_end + 2, len(beat_locations) - 1)
            if right1 == len(beat_locations) - 1:
                offset_range[_safe_beat(beat_locations, left2) :] += 1.0
            elif right2 == len(beat_locations) - 1:
                offset_range[_safe_beat(beat_locations, left2) : _safe_beat(beat_locations, right1)] += 1.0
                offset_range[_safe_beat(beat_locations, right1) :] += 0.5
            else:
                offset_range[_safe_beat(beat_locations, left2) : _safe_beat(beat_locations, right1)] += 1.0
                offset_range[
                    _safe_beat(beat_locations, right1) : min(_safe_beat(beat_locations, right2), signal_length - 1)
                ] += 0.5
            offset_range[_safe_beat(beat_locations, left3) : _safe_beat(beat_locations, left2)] += 0.5
        elif class_true == 1:
            left3 = max(af_end - 3, 0)
            left2 = max(af_end - 2, 0)
            offset_range[_safe_beat(beat_locations, left2) :] += 1.0
            offset_range[_safe_beat(beat_locations, left3) : _safe_beat(beat_locations, left2)] += 0.5

    return onset_range, offset_range


def endpoint_score(
    predicted_endpoints: np.ndarray,
    reference: Cpsc2021ReferenceInfo,
) -> float:
    if reference.class_true not in {1, 2}:
        return 0.0
    if len(predicted_endpoints) == 0:
        return 0.0
    if reference.onset_score_range is None or reference.offset_score_range is None:
        return 0.0
    score = 0.0
    for start, end in predicted_endpoints:
        score += reference.onset_score_range[int(start)]
        score += reference.offset_score_range[int(end)]
    ma = len(reference.endpoints_true)
    mr = len(predicted_endpoints)
    score *= ma / max(ma, mr)
    return float(score)


def score_record(
    record_path: str | Path,
    predicted_endpoints: list[list[int]] | np.ndarray,
) -> dict[str, float | int]:
    reference = load_reference_info(record_path)
    endpoints = np.asarray(predicted_endpoints, dtype=np.int64).reshape(-1, 2)
    class_pred = classify_prediction(endpoints, reference.signal_length)
    rhythm_score = float(RHYTHM_SCORE_MATRIX[reference.class_true, class_pred])
    endpoint_component = endpoint_score(endpoints, reference)
    total = rhythm_score + endpoint_component
    return {
        "record_score": float(total),
        "rhythm_score": rhythm_score,
        "endpoint_score": endpoint_component,
        "class_true": int(reference.class_true),
        "class_pred": int(class_pred),
    }


def score_predictions_map(
    root: str | Path,
    predictions_by_record: dict[str, list[list[int]]],
) -> dict[str, object]:
    per_record = {}
    scores = []
    for record_id, predicted_endpoints in predictions_by_record.items():
        result = score_record(Path(root) / record_id, predicted_endpoints)
        per_record[record_id] = result
        scores.append(float(result["record_score"]))
    return {
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
        "n_records": len(scores),
        "per_record": per_record,
    }


def load_prediction_json(path: str | Path) -> dict[str, list[list[int]]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if not isinstance(raw, dict):
        raise ValueError("Prediction JSON must be a mapping of record id to endpoint lists.")
    return {str(record_id): value for record_id, value in raw.items()}


__all__ = [
    "Cpsc2021ReferenceInfo",
    "classify_prediction",
    "endpoint_score",
    "generate_endpoint_score_ranges",
    "load_prediction_json",
    "load_reference_info",
    "score_predictions_map",
    "score_record",
]
