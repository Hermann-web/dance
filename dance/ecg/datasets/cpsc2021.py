from __future__ import annotations

from pathlib import Path

import numpy as np

from ..events.schema import EcgRhythmEpisode


def read_cpsc2021_record(record_path: str | Path, lead: str | int = 0) -> dict:
    """Read one CPSC2021 record through WFDB and derive AF episodes."""
    import wfdb

    record = wfdb.rdrecord(str(record_path), channels=[lead] if isinstance(lead, int) else None)
    if isinstance(lead, str):
        lowered = [name.lower() for name in record.sig_name]
        lead_idx = lowered.index(lead.strip().lower())
    else:
        lead_idx = 0
    fs = float(record.fs)
    if fs <= 0:
        raise ValueError(
            f"Invalid sampling frequency {fs} for CPSC2021 record {record_path}."
        )
    ann = wfdb.rdann(str(record_path), extension="atr")
    signal = record.p_signal[:, lead_idx].astype(np.float32)
    global_rhythm = " ".join(record.comments).strip().lower()
    episodes = episodes_from_wfdb_ann(
        ann.sample,
        list(ann.aux_note),
        fs=fs,
        signal_length=len(signal),
        global_rhythm=global_rhythm,
    )
    return {
        "record_id": Path(record_path).name,
        "signal": signal,
        "fs": fs,
        "episodes": episodes,
    }


def episodes_from_wfdb_ann(
    samples: list[int] | np.ndarray,
    aux_notes: list[str],
    *,
    fs: float,
    signal_length: int,
    global_rhythm: str = "",
    merge_gap_seconds: float = 0.0,
) -> list[EcgRhythmEpisode]:
    if signal_length <= 0:
        raise ValueError("signal_length must be positive for CPSC2021 episode parsing.")

    if "persistent atrial fibrillation" in global_rhythm:
        return [EcgRhythmEpisode(label="af_episode", onset=0, offset=signal_length - 1)]
    if "non atrial fibrillation" in global_rhythm:
        return []

    out: list[EcgRhythmEpisode] = []
    open_start: int | None = None
    for s, note in zip(samples, aux_notes):
        sample = int(s)
        if note in {"(AFIB", "(AFL"}:
            open_start = sample
            continue
        if note == "(N" and open_start is not None:
            if sample > open_start:
                out.append(
                    EcgRhythmEpisode(
                        label="af_episode",
                        onset=open_start,
                        offset=sample,
                    )
                )
            open_start = None
    if open_start is not None:
        out.append(
            EcgRhythmEpisode(
                label="af_episode",
                onset=open_start,
                offset=signal_length - 1,
            )
        )
    if not out:
        return out
    if merge_gap_seconds <= 0:
        return out
    merged = [out[0]]
    gap = int(round(merge_gap_seconds * fs))
    for ev in out[1:]:
        prev = merged[-1]
        if ev.onset - prev.offset <= gap:
            merged[-1] = EcgRhythmEpisode("af_episode", prev.onset, max(prev.offset, ev.offset))
        else:
            merged.append(ev)
    return merged
