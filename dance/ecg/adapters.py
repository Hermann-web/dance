from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

class _EcgBatchContract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    eeg: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    class_: torch.Tensor = Field(alias="class")
    channel_positions: torch.Tensor | None = None

    @model_validator(mode="after")
    def _validate(self):
        eeg, start, end, cls = self.eeg, self.start, self.end, self.class_
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
        if cls.dtype != torch.long:
            raise ValueError(f"`class` must be torch.long dtype, got {cls.dtype}.")
        cp = self.channel_positions
        if cp is not None:
            if cp.ndim != 3 or cp.shape[-1] != 2:
                raise ValueError(
                    "channel_positions must have shape (B, C, 2), "
                    f"got {tuple(cp.shape)}"
                )
            if cp.shape[:2] != eeg.shape[:2]:
                raise ValueError(
                    "channel_positions must match eeg batch/channel shape "
                    f"{tuple(eeg.shape[:2])}, got {tuple(cp.shape[:2])}"
                )
        return self


def ecg_batch_to_dance_batch(
    batch: dict[str, torch.Tensor],
    *,
    validate: bool = True,
    synthesize_channel_positions: bool = False,
) -> dict[str, torch.Tensor]:
    """Compatibility adapter into Dance.forward contract."""
    required = {"eeg", "start", "end", "class"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Missing ECG batch keys: {sorted(missing)}")
    if validate:
        try:
            _EcgBatchContract.model_validate(batch)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
    out = {
        "eeg": batch["eeg"],
        "start": batch["start"],
        "end": batch["end"],
        "class": batch["class"],
    }
    if "channel_positions" in batch:
        out["channel_positions"] = batch["channel_positions"]
    elif synthesize_channel_positions:
        bsz, channels, _ = batch["eeg"].shape
        x = torch.linspace(-1.0, 1.0, channels, device=batch["eeg"].device)
        out["channel_positions"] = torch.stack(
            [x.repeat(bsz, 1), torch.zeros((bsz, channels), device=batch["eeg"].device)],
            dim=-1,
        )
    return out
