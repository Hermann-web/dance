from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dance import Dance
from .adapters import ecg_batch_to_dance_batch
from .data import LudbDataset, ludb_collate
from .evaluation import evaluate_rhythm_events, evaluate_wave_events
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


def _extract_events_with_durations(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    durations: list[float],
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    pred_per_window: list[list[tuple]] = []
    gt_per_window: list[list[tuple]] = []
    logits_preds = preds["class"].detach().cpu()
    class_targets = targets["class"].detach().cpu()
    pred_start = preds["start"].detach().cpu()
    pred_end = preds["end"].detach().cpu()
    target_start = targets["start"].detach().cpu()
    target_end = targets["end"].detach().cpu()

    for batch_idx, window_duration in enumerate(durations):
        gt_events = []
        for start, end, cls in zip(
            target_start[batch_idx],
            target_end[batch_idx],
            class_targets[batch_idx],
        ):
            if int(cls) <= 0:
                continue
            gt_events.append(
                (
                    float(start) * window_duration,
                    float(end) * window_duration,
                    int(cls),
                )
            )
        gt_per_window.append(gt_events)

        probs = torch.softmax(logits_preds[batch_idx], dim=-1)
        scores, labels = probs.max(dim=-1)
        pred_events = []
        for query_idx, label in enumerate(labels.tolist()):
            if label == 0:
                continue
            pred_events.append(
                (
                    float(pred_start[batch_idx, query_idx]) * window_duration,
                    float(pred_end[batch_idx, query_idx]) * window_duration,
                    label,
                    float(scores[query_idx]),
                )
            )
        pred_per_window.append(pred_events)

    return pred_per_window, gt_per_window


@torch.no_grad()
def evaluate_model(
    model: Dance,
    loader: DataLoader,
    *,
    duration: float,
    task: str,
    device: str = "cpu",
) -> dict[str, float]:
    if len(loader) == 0:
        raise ValueError("evaluate_model received an empty loader.")
    model.eval()
    model.to(device)
    predicted_events = []
    target_events = []
    n_samples_per_window: list[int] = []

    for raw_batch in loader:
        batch = ecg_batch_to_dance_batch(raw_batch)
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        out = model(batch)
        signal_length = raw_batch.get("signal_length")
        fs = raw_batch.get("fs")
        if isinstance(signal_length, torch.Tensor) and isinstance(fs, torch.Tensor):
            durations = [
                float(length) / max(float(frequency), 1e-6)
                for length, frequency in zip(signal_length.tolist(), fs.tolist())
            ]
            n_samples_per_window.extend(int(value) for value in signal_length.tolist())
        else:
            durations = [duration] * batch["class"].shape[0]
            n_samples_per_window.extend([raw_batch["eeg"].shape[-1]] * batch["class"].shape[0])
        pred_batch, target_batch = _extract_events_with_durations(
            {
                "class": out["pred_class"],
                "start": out["pred_start"],
                "end": out["pred_end"],
            },
            {
                "class": batch["class"],
                "start": batch["start"],
                "end": batch["end"],
            },
            durations,
        )
        predicted_events.extend(pred_batch)
        target_events.extend(target_batch)

    if task == "ludb":
        return evaluate_wave_events(
            predicted_events,
            target_events,
            duration=duration,
            n_samples_per_window=n_samples_per_window,
        )
    if task == "cpsc2021":
        return evaluate_rhythm_events(
            predicted_events,
            target_events,
        )
    raise ValueError(f"Unsupported ECG evaluation task: {task!r}")


def save_checkpoint(
    model: Dance,
    path: str | Path,
    *,
    task: str,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "model_config": {
            "n_channels": model.n_channels,
            "n_classes": model.n_classes,
            "n_queries": model.n_queries,
            "duration": model.duration,
            "use_channel_merger": model.use_channel_merger,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[Dance, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid ECG checkpoint format: {path}")
    model = Dance(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
