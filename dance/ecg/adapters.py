from __future__ import annotations

import torch


def ecg_batch_to_dance_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compatibility adapter into Dance.forward contract."""
    required = {"eeg", "start", "end", "class"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Missing ECG batch keys: {sorted(missing)}")
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
