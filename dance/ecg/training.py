from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dance import Dance
from .adapters import ecg_batch_to_dance_batch
from .data import LudbDataset, ludb_collate


def build_ludb_loader(
    root: str | Path,
    record_ids: list[str],
    *,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    ds = LudbDataset(root=root, record_ids=record_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ludb_collate)


def train_one_epoch(
    model: Dance,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cpu",
) -> float:
    model.train()
    total = 0.0
    steps = 0
    model.to(device)
    for raw_batch in loader:
        batch = ecg_batch_to_dance_batch(raw_batch)
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)
