# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter(ABC):
    """Common interface."""

    @abstractmethod
    def __call__(
        self,
        events: pd.DataFrame,
        list_segments: tp.Callable[[pd.DataFrame, float, bool], list],
        *,
        duration: float,
        training_overlap: float = 0.0,
    ) -> tuple[list, list, list]: ...


class KFoldSplitter(Splitter):
    """Subject-grouped K-fold (one fold = test, 1/k of the rest = val)."""

    def __init__(
        self,
        *,
        k_fold: int = 5,
        fold_index: int = 0,
        valid_seed: int | None = None,
        rng_seed: int = 0,
    ) -> None:
        self.k_fold = k_fold
        self.fold_index = fold_index
        self.valid_seed = valid_seed
        self.rng_seed = rng_seed

    def __call__(
        self,
        events: pd.DataFrame,
        list_segments,
        *,
        duration: float,
        training_overlap: float = 0.0,
    ):
        subjects = np.sort(events["subject"].unique())
        rng = np.random.default_rng(self.rng_seed)
        rng.shuffle(subjects)

        folds = np.array_split(subjects, self.k_fold)
        idx = self.fold_index % self.k_fold
        test_subj = folds[idx]
        trainval_subj = np.concatenate([f for i, f in enumerate(folds) if i != idx])
        train_subj, val_subj = train_test_split(
            trainval_subj,
            test_size=0.10,
            random_state=self.valid_seed,
        )

        train_evt = events[events["subject"].isin(train_subj)]
        val_evt = events[events["subject"].isin(val_subj)]
        test_evt = events[events["subject"].isin(test_subj)]

        train_stride = duration * (1 - training_overlap)
        return (
            list_segments(train_evt, train_stride, drop_incomplete=True),
            list_segments(val_evt, duration, drop_incomplete=False),
            list_segments(test_evt, duration, drop_incomplete=False),
        )


class TuszFixedSplitter(Splitter):
    """TUSZ official train / dev / eval split read from the `split` column.

    With `debug=True`, keep only the first 3 timelines per split so that
    end-to-end smoke tests don't load the full corpus.
    """

    SPLITS = {"train": "train", "val": "dev", "test": "eval"}

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def __call__(
        self,
        events: pd.DataFrame,
        list_segments,
        *,
        duration: float,
        training_overlap: float = 0.0,
    ):
        if self.debug:
            keep = []
            for tag in self.SPLITS.values():
                keep.extend(events.loc[events["split"] == tag, "timeline"].unique()[:3])
            events = events[events["timeline"].isin(keep)]

        out = []
        for split_name, tag in self.SPLITS.items():
            stride = (
                duration * (1 - training_overlap) if split_name == "train" else duration
            )
            evt = events[events["split"] == tag]
            out.append(list_segments(evt, stride, drop_incomplete=False))
        return tuple(out)


_SPLITTERS: dict[str, type[Splitter]] = {
    "kfold": KFoldSplitter,
    "tusz_fixed": TuszFixedSplitter,
}


def build_splitter(name: str, **kwargs) -> Splitter:
    """Instantiate a splitter by registry name (used by YAML configs)."""
    try:
        cls = _SPLITTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown splitter {name!r}. Available: {sorted(_SPLITTERS)}"
        ) from exc
    return cls(**kwargs)
