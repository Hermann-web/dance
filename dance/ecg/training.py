from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dance import Dance
from .adapters import ecg_batch_to_dance_batch
from .data import LudbDataset, ludb_collate
from .rhythm_data import Cpsc2021Dataset, cpsc2021_collate


def build_ludb_loader(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    if not record_ids:
        raise ValueError("build_ludb_loader requires at least one record id.")
    ds = LudbDataset(root=root, record_ids=record_ids, lead=lead)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ludb_collate)


def build_cpsc2021_loader(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    if not record_ids:
        raise ValueError("build_cpsc2021_loader requires at least one record id.")
    ds = Cpsc2021Dataset(root=root, record_ids=record_ids, lead=lead)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cpsc2021_collate,
    )


def train_one_epoch(
    model: Dance,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cpu",
) -> float:
    if len(loader) == 0:
        raise ValueError("train_one_epoch received an empty loader.")
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
