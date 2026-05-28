from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .classifier import run_cpsc2021_logreg_baseline
from .datasets.cpsc2021 import read_cpsc2021_record


def infer_cpsc2021_subject_id(record_id: str | Path) -> str:
    stem = Path(record_id).name
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def load_cpsc2021_record_labels(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
) -> list[int]:
    labels: list[int] = []
    for record_id in record_ids:
        sample = read_cpsc2021_record(Path(root) / record_id, lead=lead)
        labels.append(int(len(sample["episodes"]) > 0))
    return labels


def build_cpsc2021_group_splits(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
    n_splits: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int = 0,
) -> list[dict]:
    if len(record_ids) < 2:
        raise ValueError("build_cpsc2021_group_splits requires at least two records.")
    groups = [infer_cpsc2021_subject_id(record_id) for record_id in record_ids]
    if len(set(groups)) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct subject groups, got {len(set(groups))}."
        )
    labels = load_cpsc2021_record_labels(root, record_ids, lead=lead)
    indices = np.arange(len(record_ids))
    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        if shuffle:
            rng = np.random.default_rng(random_state)
            order = rng.permutation(indices)
            record_ids = [record_ids[i] for i in order]
            groups = [groups[i] for i in order]
            labels = [labels[i] for i in order]
            indices = np.arange(len(record_ids))
        splitter = GroupKFold(n_splits=n_splits)

    folds: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(indices, labels, groups),
        start=1,
    ):
        train_records = [record_ids[i] for i in train_idx]
        test_records = [record_ids[i] for i in test_idx]
        train_groups = sorted({groups[i] for i in train_idx})
        test_groups = sorted({groups[i] for i in test_idx})
        folds.append(
            {
                "fold": fold_idx,
                "train_records": train_records,
                "test_records": test_records,
                "train_subjects": train_groups,
                "test_subjects": test_groups,
                "train_positive_records": int(sum(labels[i] for i in train_idx)),
                "test_positive_records": int(sum(labels[i] for i in test_idx)),
            }
        )
    return folds


def run_cpsc2021_logreg_cv(
    *,
    root: str | Path,
    record_ids: list[str],
    lead: str | int = 0,
    window_duration_s: float = 30.0,
    window_stride_s: float | None = None,
    n_splits: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int = 0,
    c: float = 1.0,
    max_iter: int = 1000,
    threshold: float = 0.5,
) -> dict:
    folds = build_cpsc2021_group_splits(
        root,
        record_ids,
        lead=lead,
        n_splits=n_splits,
        stratified=stratified,
        shuffle=shuffle,
        random_state=random_state,
    )
    fold_results: list[dict] = []
    metric_names = (
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auroc",
        "average_precision",
    )
    for fold in folds:
        result = run_cpsc2021_logreg_baseline(
            root=root,
            train_record_ids=fold["train_records"],
            test_record_ids=fold["test_records"],
            lead=lead,
            window_duration_s=window_duration_s,
            window_stride_s=window_stride_s,
            c=c,
            max_iter=max_iter,
            threshold=threshold,
        )
        fold_results.append(
            {
                **fold,
                **result,
            }
        )

    aggregate = {}
    for metric_name in metric_names:
        values = np.asarray([fold[metric_name] for fold in fold_results], dtype=np.float64)
        aggregate[f"{metric_name}_mean"] = float(np.nanmean(values))
        aggregate[f"{metric_name}_std"] = float(np.nanstd(values))

    return {
        "n_splits": n_splits,
        "splitter": "StratifiedGroupKFold" if stratified else "GroupKFold",
        "random_state": random_state,
        "window_duration_s": window_duration_s,
        "window_stride_s": window_stride_s,
        "folds": fold_results,
        **aggregate,
    }


__all__ = [
    "build_cpsc2021_group_splits",
    "infer_cpsc2021_subject_id",
    "load_cpsc2021_record_labels",
    "run_cpsc2021_logreg_cv",
]
