from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ._batching import collate_interval_batch
from .datasets.cpsc2021 import read_cpsc2021_record
from .events import RHYTHM_CLASS_TO_ID


class Cpsc2021Dataset(Dataset):
    def __init__(self, root: str | Path, record_ids: list[str], lead: str | int = 0):
        self.root = Path(root)
        self.record_ids = record_ids
        self.lead = lead

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> dict:
        rid = self.record_ids[idx]
        sample = read_cpsc2021_record(self.root / rid, lead=self.lead)
        signal = torch.from_numpy(sample["signal"]).unsqueeze(0)
        starts = np.array([e.onset for e in sample["episodes"]], dtype=np.float32)
        ends = np.array([e.offset for e in sample["episodes"]], dtype=np.float32)
        classes = np.array(
            [RHYTHM_CLASS_TO_ID[e.label] for e in sample["episodes"]],
            dtype=np.int64,
        )
        return {
            "record_id": sample["record_id"],
            "eeg": signal,
            "fs": sample["fs"],
            "event_start": starts,
            "event_end": ends,
            "event_class": classes,
        }


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
