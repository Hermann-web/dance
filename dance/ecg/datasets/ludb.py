from __future__ import annotations

from pathlib import Path

import numpy as np
from ..events.schema import EcgWaveEvent, WAVE_ANN_TO_CLASS


def read_ludb_record(record_path: str | Path, lead: str | int = 0) -> dict:
    """Read one LUDB record and convert delineation annotations to events."""
    import wfdb

    record = wfdb.rdrecord(str(record_path), channels=[lead] if isinstance(lead, int) else None)
    if isinstance(lead, str):
        lead_idx = record.sig_name.index(lead)
    else:
        lead_idx = 0
    ann = wfdb.rdann(str(record_path), extension="atr")
    signal = record.p_signal[:, lead_idx].astype(np.float32)
    events = _events_from_wfdb_ann(ann.sample, list(ann.symbol))
    return {
        "record_id": Path(record_path).name,
        "signal": signal,
        "fs": float(record.fs),
        "events": events,
    }


def _events_from_wfdb_ann(samples: list[int] | np.ndarray, symbols: list[str]) -> list[EcgWaveEvent]:
    events: list[EcgWaveEvent] = []
    stack: dict[str, int] = {}
    for sample, sym in zip(samples, symbols):
        if sym.startswith("("):
            stack[sym[1:]] = int(sample)
            continue
        if sym.startswith(")") and sym[1:] in stack:
            key = sym[1:]
            label = WAVE_ANN_TO_CLASS.get(key)
            if label is None:
                stack.pop(key, None)
                continue
            start = stack.pop(key)
            end = int(sample)
            if end > start:
                events.append(EcgWaveEvent(label=label, onset=start, offset=end))
    return events
