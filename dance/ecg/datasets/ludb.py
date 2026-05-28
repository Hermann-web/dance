from __future__ import annotations

from pathlib import Path

import numpy as np

from ..events.schema import WAVE_ANN_TO_CLASS, EcgWaveEvent


def _resolve_ludb_lead(record_path: str | Path, lead: str | int) -> tuple[int, str]:
    import wfdb

    header = wfdb.rdheader(str(record_path))
    sig_names = list(header.sig_name)
    if isinstance(lead, int):
        if lead < 0 or lead >= len(sig_names):
            raise IndexError(
                f"Lead index {lead} out of range for LUDB record {record_path}. "
                f"Available leads: 0..{len(sig_names) - 1}"
            )
        lead_idx = lead
    else:
        wanted = lead.strip().lower()
        lowered = [name.lower() for name in sig_names]
        if wanted not in lowered:
            raise ValueError(
                f"Lead {lead!r} not found in LUDB record {record_path}. "
                f"Available leads: {sig_names}"
            )
        lead_idx = lowered.index(wanted)
    return lead_idx, sig_names[lead_idx].lower()


def read_ludb_record(record_path: str | Path, lead: str | int = 0) -> dict:
    """Read one LUDB record and convert delineation annotations to events."""
    import wfdb

    lead_idx, ann_extension = _resolve_ludb_lead(record_path, lead)
    record = wfdb.rdrecord(str(record_path), channels=[lead_idx])
    ann = wfdb.rdann(str(record_path), extension=ann_extension)
    fs = float(record.fs)
    if fs <= 0:
        raise ValueError(f"Invalid sampling frequency {fs} for LUDB record {record_path}.")
    signal = record.p_signal[:, 0].astype(np.float32)
    events = _events_from_wfdb_ann(ann.sample, list(ann.symbol))
    return {
        "record_id": Path(record_path).name,
        "signal": signal,
        "fs": fs,
        "events": events,
    }


def _events_from_wfdb_ann(samples: list[int] | np.ndarray, symbols: list[str]) -> list[EcgWaveEvent]:
    events: list[EcgWaveEvent] = []
    i = 0
    while i + 2 < len(samples):
        start_sym, peak_sym, end_sym = symbols[i : i + 3]
        if start_sym == "(" and end_sym == ")" and peak_sym in WAVE_ANN_TO_CLASS:
            start = int(samples[i])
            end = int(samples[i + 2])
            if end > start:
                events.append(
                    EcgWaveEvent(
                        label=WAVE_ANN_TO_CLASS[peak_sym],
                        onset=start,
                        offset=end,
                    )
                )
            i += 3
            continue
        i += 1
    return events
