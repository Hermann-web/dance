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
    duration: float | None = None,
    stride: float | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    sampler=None,
) -> DataLoader:
    if not record_ids:
        raise ValueError("build_ludb_loader requires at least one record id.")
    ds = LudbDataset(
        root=root,
        record_ids=record_ids,
        lead=lead,
        window_duration_s=duration,
        window_stride_s=stride,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=ludb_collate,
    )


def build_cpsc2021_loader(
    root: str | Path,
    record_ids: list[str],
    *,
    lead: str | int = 0,
    duration: float | None = None,
    stride: float | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    use_weighted_sampler: bool = False,
    positive_weight: float = 5.0,
    negative_weight: float = 1.0,
) -> DataLoader:
    if not record_ids:
        raise ValueError("build_cpsc2021_loader requires at least one record id.")
    ds = Cpsc2021Dataset(
        root=root,
        record_ids=record_ids,
        lead=lead,
        window_duration_s=duration,
        window_stride_s=stride,
    )
    sampler = None
    if use_weighted_sampler:
        from .rhythm_data import build_rhythm_weighted_sampler

        sampler = build_rhythm_weighted_sampler(
            ds,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
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
