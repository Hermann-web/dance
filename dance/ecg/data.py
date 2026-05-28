from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
    max_len = max(item["eeg"].shape[-1] for item in batch)
    max_events = max(len(item["event_class"]) for item in batch) if batch else 0
    eeg = torch.zeros((len(batch), 1, max_len), dtype=torch.float32)
    start = torch.zeros((len(batch), max_events), dtype=torch.float32)
    end = torch.zeros((len(batch), max_events), dtype=torch.float32)
    cls = torch.zeros((len(batch), max_events), dtype=torch.long)
    fs = torch.tensor([item["fs"] for item in batch], dtype=torch.float32)
    ids: list[str] = []
    for i, item in enumerate(batch):
        sig = item["eeg"]
        T = sig.shape[-1]
        eeg[i, :, :T] = sig
        n = len(item["event_class"])
        if n:
            start[i, :n] = torch.from_numpy(item["event_start"]) / float(T)
            end[i, :n] = torch.from_numpy(item["event_end"]) / float(T)
            cls[i, :n] = torch.from_numpy(item["event_class"])
        ids.append(item["record_id"])
    return {
        "record_id": ids,
        "eeg": eeg,
        "fs": fs,
        "start": start.clamp(0.0, 1.0),
        "end": end.clamp(0.0, 1.0),
        "class": cls,
    }
