from __future__ import annotations

import torch


def ecg_batch_to_dance_batch(
    batch: dict[str, torch.Tensor],
    *,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Compatibility adapter into Dance.forward contract."""
    required = {"eeg", "start", "end", "class"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Missing ECG batch keys: {sorted(missing)}")
    if validate:
        eeg = batch["eeg"]
        start = batch["start"]
        end = batch["end"]
        cls = batch["class"]
        if eeg.ndim != 3:
            raise ValueError(f"`eeg` must have shape (B, C, T), got {tuple(eeg.shape)}")
        if start.shape != end.shape or start.shape != cls.shape:
            raise ValueError(
                "`start`, `end`, and `class` must share shape (B, max_events); "
                f"got {tuple(start.shape)}, {tuple(end.shape)}, {tuple(cls.shape)}"
            )
        if torch.any(end < start):
            raise ValueError("Found event with end < start in ECG batch.")
        if torch.any((start < 0) | (start > 1) | (end < 0) | (end > 1)):
            raise ValueError("`start`/`end` must be normalized to [0, 1].")
    out = {
        "eeg": batch["eeg"],
        "start": batch["start"],
        "end": batch["end"],
        "class": batch["class"],
    }
    if "channel_positions" in batch:
        out["channel_positions"] = batch["channel_positions"]
    else:
        bsz, channels, _ = batch["eeg"].shape
        x = torch.linspace(-1.0, 1.0, channels, device=batch["eeg"].device)
        out["channel_positions"] = torch.stack(
            [x.repeat(bsz, 1), torch.zeros((bsz, channels), device=batch["eeg"].device)],
            dim=-1,
        )
    return out
