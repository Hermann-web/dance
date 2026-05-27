# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Unit tests for KFoldSplitter (the splitter used by every dataset except TUSZ)."""

from __future__ import annotations

import pandas as pd

from dance.training.splitters import KFoldSplitter


def _events_for(n_subjects: int) -> pd.DataFrame:
    """One Eeg timeline + one Stimulus event per fake subject."""
    rows = []
    for subj in range(n_subjects):
        rows.append(
            {
                "timeline": f"t{subj}",
                "subject": str(subj),
                "type": "Eeg",
                "start": 0.0,
                "duration": 10.0,
            }
        )
        rows.append(
            {
                "timeline": f"t{subj}",
                "subject": str(subj),
                "type": "Stimulus",
                "start": 1.0,
                "duration": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _list_segments(events, stride, drop_incomplete):
    """Stand-in for ns.segments.list_segments — yields one tuple per Eeg row."""
    return [(e.timeline, e.start, stride) for e in events.itertuples() if e.type == "Eeg"]


def test_kfold_partitions_subjects_into_disjoint_sets():
    """No subject appears in more than one split, and the test fold has the
    expected size (n_subjects / k_fold)."""
    splitter = KFoldSplitter(k_fold=5, fold_index=0, valid_seed=0)
    train, val, test = splitter(
        _events_for(n_subjects=20),
        _list_segments,
        duration=8.0,
        training_overlap=0.0,
    )
    train_subj = {tl for tl, _, _ in train}
    val_subj = {tl for tl, _, _ in val}
    test_subj = {tl for tl, _, _ in test}
    assert train_subj.isdisjoint(test_subj)
    assert train_subj.isdisjoint(val_subj)
    assert val_subj.isdisjoint(test_subj)
    assert len(test_subj) == 20 // 5
