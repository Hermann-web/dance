from __future__ import annotations

import numpy as np
import torch


def collate_interval_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("collate_interval_batch called with empty batch")
    max_len = max(item["eeg"].shape[-1] for item in batch)
    max_events = max(len(item["event_class"]) for item in batch)
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
            starts = np.asarray(item["event_start"], dtype=np.float32)
            ends = np.asarray(item["event_end"], dtype=np.float32)
            classes = np.asarray(item["event_class"], dtype=np.int64)
            start[i, :n] = torch.from_numpy(starts) / float(T)
            end[i, :n] = torch.from_numpy(ends) / float(T)
            cls[i, :n] = torch.from_numpy(classes)
        ids.append(item["record_id"])
    return {
        "record_id": ids,
        "eeg": eeg,
        "fs": fs,
        "start": start.clamp(0.0, 1.0),
        "end": end.clamp(0.0, 1.0),
        "class": cls,
    }
