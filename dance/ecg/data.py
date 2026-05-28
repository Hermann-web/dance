from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ._batching import collate_interval_batch
from .datasets.ludb import read_ludb_record
from .events.schema import WAVE_CLASS_TO_ID


class LudbDataset(Dataset):
    def __init__(self, root: str | Path, record_ids: list[str], lead: str | int = 0):
        self.root = Path(root)
        self.record_ids = record_ids
        self.lead = lead

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> dict:
        rid = self.record_ids[idx]
        sample = read_ludb_record(self.root / rid, lead=self.lead)
        signal = torch.from_numpy(sample["signal"]).unsqueeze(0)
        starts = np.array([e.onset for e in sample["events"]], dtype=np.float32)
        ends = np.array([e.offset for e in sample["events"]], dtype=np.float32)
        classes = np.array([WAVE_CLASS_TO_ID[e.label] for e in sample["events"]], dtype=np.int64)
        return {
            "record_id": sample["record_id"],
            "eeg": signal,
            "fs": sample["fs"],
            "event_start": starts,
            "event_end": ends,
            "event_class": classes,
        }


def ludb_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return collate_interval_batch(batch)
