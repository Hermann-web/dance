from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ._batching import collate_interval_batch
from .datasets.ludb import read_ludb_record
from .events.schema import WAVE_CLASS_TO_ID


class LudbDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        record_ids: list[str],
        lead: str | int = 0,
        *,
        window_duration_s: float | None = None,
        window_stride_s: float | None = None,
    ):
        self.root = Path(root)
        self.record_ids = record_ids
        self.lead = lead
        self.window_duration_s = window_duration_s
        self.window_stride_s = window_stride_s
        self._windows = self._build_windows()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        rid, start_sample, end_sample = self._windows[idx]
        sample = read_ludb_record(self.root / rid, lead=self.lead)
        signal = torch.from_numpy(sample["signal"][start_sample:end_sample]).unsqueeze(0)
        clipped_events = _clip_events(sample["events"], start_sample, end_sample)
        starts = np.array([e.onset for e in clipped_events], dtype=np.float32)
        ends = np.array([e.offset for e in clipped_events], dtype=np.float32)
        classes = np.array(
            [WAVE_CLASS_TO_ID[e.label] for e in clipped_events],
            dtype=np.int64,
        )
        return {
            "record_id": f"{sample['record_id']}:{start_sample}-{end_sample}",
            "eeg": signal,
            "fs": sample["fs"],
            "event_start": starts,
            "event_end": ends,
            "event_class": classes,
        }

    def _build_windows(self) -> list[tuple[str, int, int]]:
        windows: list[tuple[str, int, int]] = []
        for rid in self.record_ids:
            sample = read_ludb_record(self.root / rid, lead=self.lead)
            total = len(sample["signal"])
            window = _resolve_window_samples(
                total=total,
                fs=sample["fs"],
                duration_s=self.window_duration_s,
                stride_s=self.window_stride_s,
            )
            windows.extend((rid, start, end) for start, end in window)
        return windows


def _resolve_window_samples(
    *,
    total: int,
    fs: float,
    duration_s: float | None,
    stride_s: float | None,
) -> list[tuple[int, int]]:
    if duration_s is None:
        return [(0, total)]
    window = max(1, int(round(duration_s * fs)))
    if window >= total:
        return [(0, total)]
    stride = max(1, int(round((stride_s or duration_s) * fs)))
    starts = list(range(0, total - window + 1, stride))
    last_start = total - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window) for start in starts]


def _clip_events(events, start_sample: int, end_sample: int):
    clipped = []
    for ev in events:
        if ev.offset <= start_sample or ev.onset >= end_sample:
            continue
        onset = max(ev.onset, start_sample) - start_sample
        offset = min(ev.offset, end_sample) - start_sample
        if offset > onset:
            clipped.append(type(ev)(label=ev.label, onset=onset, offset=offset))
    return clipped


def ludb_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return collate_interval_batch(batch)
