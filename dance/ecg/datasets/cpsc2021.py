from __future__ import annotations

from pathlib import Path

import numpy as np

from ..events.schema import EcgRhythmEpisode


def read_cpsc2021_record(record_path: str | Path, lead: str | int = 0) -> dict:
    """Read one CPSC2021 record through WFDB and derive AF episodes."""
    import wfdb

    record = wfdb.rdrecord(str(record_path), channels=[lead] if isinstance(lead, int) else None)
    if isinstance(lead, str):
        lead_idx = record.sig_name.index(lead)
    else:
        lead_idx = 0
    ann = wfdb.rdann(str(record_path), extension="atr")
    signal = record.p_signal[:, lead_idx].astype(np.float32)
    episodes = episodes_from_wfdb_ann(ann.sample, list(ann.symbol), fs=float(record.fs))
    return {
        "record_id": Path(record_path).name,
        "signal": signal,
        "fs": float(record.fs),
        "episodes": episodes,
    }


def episodes_from_wfdb_ann(
    samples: list[int] | np.ndarray,
    symbols: list[str],
    *,
    fs: float,
    merge_gap_seconds: float = 0.0,
) -> list[EcgRhythmEpisode]:
    out: list[EcgRhythmEpisode] = []
    open_start: int | None = None
    for s, sym in zip(samples, symbols):
        sample = int(s)
        if sym == "(AFIB":
            open_start = sample
            continue
        if sym == ")AFIB" and open_start is not None:
            if sample > open_start:
                out.append(
                    EcgRhythmEpisode(
                        label="af_episode",
                        onset=open_start,
                        offset=sample,
                    )
                )
            open_start = None
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
