from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ._batching import collate_interval_batch
from .data import _clip_events, _resolve_window_samples
from .datasets.cpsc2021 import read_cpsc2021_record
from .events import RHYTHM_CLASS_TO_ID


class Cpsc2021Dataset(Dataset):
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
        sample = read_cpsc2021_record(self.root / rid, lead=self.lead)
        signal = torch.from_numpy(sample["signal"][start_sample:end_sample]).unsqueeze(0)
        clipped_events = _clip_events(sample["episodes"], start_sample, end_sample)
        starts = np.array([e.onset for e in clipped_events], dtype=np.float32)
        ends = np.array([e.offset for e in clipped_events], dtype=np.float32)
        classes = np.array(
            [RHYTHM_CLASS_TO_ID[e.label] for e in clipped_events],
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
            sample = read_cpsc2021_record(self.root / rid, lead=self.lead)
            total = len(sample["signal"])
            window = _resolve_window_samples(
                total=total,
                fs=sample["fs"],
                duration_s=self.window_duration_s,
                stride_s=self.window_stride_s,
            )
            windows.extend((rid, start, end) for start, end in window)
        return windows


def cpsc2021_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return collate_interval_batch(batch)


def build_rhythm_weighted_sampler(
    dataset: Cpsc2021Dataset,
    *,
    positive_weight: float = 5.0,
    negative_weight: float = 1.0,
) -> WeightedRandomSampler:
    weights: list[float] = []
    for i in range(len(dataset)):
        item = dataset[i]
        has_episode = len(item["event_class"]) > 0
        weights.append(positive_weight if has_episode else negative_weight)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
