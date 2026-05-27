"""MOABB BI2014b -> DANCE batch dict adapter (pure moabb + mne + torch).

The model expects a per-window dict with:

    eeg                 : (B, n_channels, T) float
    channel_positions   : (B, n_channels, 2) float, normalised (x, y) electrode coords
    start, end          : (B, max_events) float in [0, 1] (window-relative)
    class               : (B, max_events) long, 0 = padding/no-event, 1 = NonTarget, 2 = Target
    dense               : (B, num_latents) long, per-token target class id

This file is the bridge between MOABB's raw MNE recordings and that
schema. Drop it in any external project that wants to train DANCE on a
MOABB P300 dataset; swap `_DATASET_CLS` to use a different one.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import numpy as np
import torch
from moabb.datasets import BI2014b
from torch.utils.data import Dataset

_DATASET_CLS = BI2014b
DURATION_S = 32.0  # 32-second windows (DANCE paper config)
MAX_EVENTS = 150  # = n_queries; per-window stimulus count is well below
N_CLASSES = 3  # 0 = padding/background, 1 = NonTarget, 2 = Target

# MOABB downloads to $MNE_DATA on first use; override here if you want
# a project-local cache, e.g. moabb.set_download_dir("./data").


@dataclasses.dataclass(frozen=True)
class Window:
    """One (eeg, events) chunk feeding a single batch row."""

    eeg: np.ndarray  # (n_channels, T) at the raw sample rate
    channel_positions: np.ndarray  # (n_channels, 2) normalised (x, y) in [0, 1]
    starts: np.ndarray  # (n_events,) seconds from window start
    ends: np.ndarray  # (n_events,) seconds from window start
    classes: np.ndarray  # (n_events,) int class ids in [1, n_classes)


def _channel_positions(raw) -> np.ndarray:
    """Read the (n_channels, 2) normalised xy electrode layout from an MNE raw."""
    xyz = np.array([ch["loc"][:3] for ch in raw.info["chs"]])  # (n_ch, 3)
    xy = xyz[:, :2]
    # Normalise to [0, 1] in each axis (DANCE's ChannelMerger expects [0, 1]).
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    return (xy - mn) / np.maximum(mx - mn, 1e-9)


def _stimulus_events(raw, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract (onset_seconds, class_id) for every Target / NonTarget event."""
    onsets, classes = [], []
    for ann in raw.annotations:
        descr = ann["description"]
        cls = {"NonTarget": 1, "Target": 2}.get(descr)
        if cls is None:
            continue
        onsets.append(ann["onset"])
        classes.append(cls)
    return np.array(onsets), np.array(classes, dtype=np.int64)


def iter_windows(subjects: list[int]) -> Iterator[Window]:
    """Yield non-overlapping 32-second windows across the given subjects."""
    ds = _DATASET_CLS()
    sessions = ds.get_data(subjects=subjects)
    for _subj, sess_dict in sessions.items():
        for _sess, runs in sess_dict.items():
            for _run, raw in runs.items():
                sfreq = raw.info["sfreq"]
                n_samples = raw.n_times
                window_samples = int(DURATION_S * sfreq)
                positions = _channel_positions(raw)
                onsets, classes = _stimulus_events(raw, sfreq)
                # Plain EEG matrix; standardise per channel for numerical sanity.
                eeg = raw.get_data()  # (n_ch, T)
                eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (
                    eeg.std(axis=1, keepdims=True) + 1e-9
                )
                for w_start in range(0, n_samples - window_samples + 1, window_samples):
                    w_end_s = (w_start + window_samples) / sfreq
                    w_start_s = w_start / sfreq
                    mask = (onsets >= w_start_s) & (onsets < w_end_s)
                    yield Window(
                        eeg=eeg[:, w_start : w_start + window_samples].astype(np.float32),
                        channel_positions=positions.astype(np.float32),
                        starts=(onsets[mask] - w_start_s).astype(np.float32),
                        ends=(onsets[mask] - w_start_s + 1.0).astype(
                            np.float32
                        ),  # P300 stimuli last ~1 s
                        classes=classes[mask],
                    )


class DanceBI2014bDataset(Dataset):
    """In-memory Dataset of pre-windowed BI2014b examples."""

    def __init__(self, subjects: list[int]) -> None:
        self.windows: list[Window] = list(iter_windows(subjects))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Window:
        return self.windows[idx]


def collate(batch: list[Window]) -> dict[str, torch.Tensor]:
    """Stack a list of `Window`s into the DANCE batch dict.

    Per-event tensors are zero-padded to `MAX_EVENTS` (= n_queries).
    The per-timestep `dense` target is derived from (start, end, class)
    inside `Dance.forward`, so we do not assemble it here.
    """
    B = len(batch)

    eeg = torch.from_numpy(np.stack([w.eeg for w in batch]))
    positions = torch.from_numpy(np.stack([w.channel_positions for w in batch]))
    starts = torch.zeros(B, MAX_EVENTS, dtype=torch.float32)
    ends = torch.zeros(B, MAX_EVENTS, dtype=torch.float32)
    classes = torch.zeros(B, MAX_EVENTS, dtype=torch.long)
    for i, w in enumerate(batch):
        n = min(len(w.starts), MAX_EVENTS)
        starts[i, :n] = torch.from_numpy(w.starts[:n] / DURATION_S)  # normalise to [0, 1]
        ends[i, :n] = torch.from_numpy(np.minimum(w.ends[:n], DURATION_S) / DURATION_S)
        classes[i, :n] = torch.from_numpy(w.classes[:n])
    return {
        "eeg": eeg,
        "channel_positions": positions,
        "start": starts,
        "end": ends,
        "class": classes,
    }
