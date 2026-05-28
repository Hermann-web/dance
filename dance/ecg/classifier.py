from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import _resolve_window_samples
from .datasets.cpsc2021 import read_cpsc2021_record

FEATURE_NAMES = (
    "signal_mean",
    "signal_std",
    "signal_rms",
    "signal_abs_mean",
    "signal_iqr",
    "signal_peak_to_peak",
    "signal_line_length",
    "signal_zero_crossing_rate",
    "r_peak_rate_hz",
    "rr_mean",
    "rr_std",
    "rr_rmssd",
    "rr_pnn50",
    "rr_median_hr_bpm",
)


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _bandpass_for_qrs(signal: np.ndarray, fs: float) -> np.ndarray:
    if signal.size < 9 or fs <= 0:
        return signal.astype(np.float64, copy=False)
    nyquist = fs / 2.0
    low = max(0.5, min(5.0, nyquist * 0.45))
    high = min(20.0, nyquist * 0.95)
    if high <= low:
        return signal.astype(np.float64, copy=False)
    b, a = butter(2, [low / nyquist, high / nyquist], btype="bandpass")
    try:
        return filtfilt(b, a, signal.astype(np.float64, copy=False))
    except ValueError:
        return signal.astype(np.float64, copy=False)


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    filtered = _bandpass_for_qrs(signal, fs)
    centered = filtered - np.median(filtered)
    sigma = float(np.std(centered))
    distance = max(1, int(round(0.25 * fs)))
    prominence = max(0.0, 0.5 * sigma)
    pos_peaks, _ = find_peaks(centered, distance=distance, prominence=prominence)
    neg_peaks, _ = find_peaks(-centered, distance=distance, prominence=prominence)
    peaks = neg_peaks if len(neg_peaks) > len(pos_peaks) else pos_peaks
    return np.asarray(peaks, dtype=np.int64)


def extract_ecg_rr_features(signal: Sequence[float] | np.ndarray, fs: float) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("extract_ecg_rr_features requires a non-empty signal.")
    centered = array - np.median(array)
    rms = np.sqrt(np.mean(centered**2))
    diff = np.diff(centered)
    peaks = detect_r_peaks(array, fs)
    rr = np.diff(peaks) / float(fs) if len(peaks) >= 2 and fs > 0 else np.array([], dtype=np.float64)
    rr_diff = np.diff(rr) if len(rr) >= 2 else np.array([], dtype=np.float64)
    hr = 60.0 / rr if len(rr) else np.array([], dtype=np.float64)

    features = np.array(
        [
            np.mean(centered),
            np.std(centered),
            rms,
            np.mean(np.abs(centered)),
            np.subtract(*np.percentile(centered, [75, 25])),
            np.ptp(centered),
            np.mean(np.abs(diff)) if diff.size else 0.0,
            np.mean(np.diff(np.signbit(centered)).astype(np.float64)) if centered.size > 1 else 0.0,
            len(peaks) / max(array.size / float(fs), 1e-6) if fs > 0 else 0.0,
            np.mean(rr) if len(rr) else 0.0,
            np.std(rr) if len(rr) else 0.0,
            np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) else 0.0,
            np.mean(np.abs(rr_diff) > 0.05) if len(rr_diff) else 0.0,
            np.median(hr) if len(hr) else 0.0,
        ],
        dtype=np.float64,
    )
    return np.array([_safe_float(value) for value in features], dtype=np.float64)


def _window_has_af(episodes, start_sample: int, end_sample: int) -> bool:
    for episode in episodes:
        if episode.offset <= start_sample or episode.onset >= end_sample:
            continue
        return True
    return False


def build_cpsc2021_classification_table(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
    window_duration_s: float = 30.0,
    window_stride_s: float | None = None,
) -> dict[str, np.ndarray | list[str] | list[tuple[int, int]]]:
    if not record_ids:
        raise ValueError("build_cpsc2021_classification_table requires at least one record id.")

    features: list[np.ndarray] = []
    labels: list[int] = []
    window_ids: list[str] = []
    windows: list[tuple[int, int]] = []

    for record_id in record_ids:
        sample = read_cpsc2021_record(Path(root) / record_id, lead=lead)
        signal = sample["signal"]
        fs = float(sample["fs"])
        episodes = sample["episodes"]
        for start_sample, end_sample in _resolve_window_samples(
            total=len(signal),
            fs=fs,
            duration_s=window_duration_s,
            stride_s=window_stride_s,
        ):
            window_signal = signal[start_sample:end_sample]
            features.append(extract_ecg_rr_features(window_signal, fs))
            labels.append(int(_window_has_af(episodes, start_sample, end_sample)))
            window_ids.append(sample["record_id"])
            windows.append((start_sample, end_sample))

    if not features:
        raise ValueError("No CPSC2021 classification windows were produced.")

    return {
        "X": np.vstack(features),
        "y": np.asarray(labels, dtype=np.int64),
        "record_ids": window_ids,
        "windows": windows,
        "feature_names": list(FEATURE_NAMES),
    }


def evaluate_binary_classifier(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)
    tp = int(np.logical_and(pred == 1, truth == 1).sum())
    tn = int(np.logical_and(pred == 0, truth == 0).sum())
    fp = int(np.logical_and(pred == 1, truth == 0).sum())
    fn = int(np.logical_and(pred == 0, truth == 1).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / max(len(truth), 1)
    if np.isnan(precision) or np.isnan(sensitivity) or (precision + sensitivity) <= 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)

    if len(np.unique(truth)) < 2:
        auroc = float("nan")
        average_precision = float("nan")
    else:
        auroc = float(roc_auc_score(truth, prob))
        average_precision = float(average_precision_score(truth, prob))

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auroc": auroc,
        "average_precision": average_precision,
    }


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    if len(np.unique(y)) < 2:
        raise ValueError("Logistic regression baseline requires both AF-positive and AF-negative training windows.")
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=max_iter,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def run_cpsc2021_logreg_baseline(
    *,
    root: str | Path,
    train_record_ids: list[str],
    test_record_ids: list[str],
    lead: str | int = 0,
    window_duration_s: float = 30.0,
    window_stride_s: float | None = None,
    c: float = 1.0,
    max_iter: int = 1000,
    threshold: float = 0.5,
) -> dict[str, float | int | list[str]]:
    if not train_record_ids:
        raise ValueError("train_record_ids must contain at least one record id.")
    if not test_record_ids:
        raise ValueError("test_record_ids must contain at least one record id.")

    train_table = build_cpsc2021_classification_table(
        root,
        train_record_ids,
        lead=lead,
        window_duration_s=window_duration_s,
        window_stride_s=window_stride_s,
    )
    test_table = build_cpsc2021_classification_table(
        root,
        test_record_ids,
        lead=lead,
        window_duration_s=window_duration_s,
        window_stride_s=window_stride_s,
    )

    model = fit_logistic_regression(
        train_table["X"],
        train_table["y"],
        c=c,
        max_iter=max_iter,
    )
    test_prob = model.predict_proba(test_table["X"])[:, 1]
    metrics = evaluate_binary_classifier(
        test_table["y"],
        test_prob,
        threshold=threshold,
    )
    positive_train = int(np.sum(train_table["y"]))
    positive_test = int(np.sum(test_table["y"]))
    return {
        **metrics,
        "n_train_windows": int(len(train_table["y"])),
        "n_test_windows": int(len(test_table["y"])),
        "n_train_positive_windows": positive_train,
        "n_test_positive_windows": positive_test,
        "feature_names": list(FEATURE_NAMES),
    }


__all__ = [
    "FEATURE_NAMES",
    "build_cpsc2021_classification_table",
    "detect_r_peaks",
    "evaluate_binary_classifier",
    "extract_ecg_rr_features",
    "fit_logistic_regression",
    "run_cpsc2021_logreg_baseline",
]
