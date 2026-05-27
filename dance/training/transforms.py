# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import numpy as np
import pandas as pd
from neuralset.events.study import EventsTransform

TUSZ_BAD_TIMELINES: tuple[str, ...] = (
    "5c6f7278b9_Shah2018_subject-aaa_5c6f7278b9_figuration-02_tcp_le",
)


class SeizurePreprocessor(EventsTransform):
    """Clean up the TUSZ Seizure event table for the 4-class classification head.

    1. Deduplicate per-channel Seizure annotations to one event per onset
       (TUSZ ships them as one row per affected channel).
    2. Drop background ("bckg") rows and generic "seiz" rows that carry
       no seizure subtype.
    3. Drop a hard-coded list of known-bad timelines (TUSZ_BAD_TIMELINES).
    4. Drop Seizures with > 99% temporal overlap with the previous
       Seizure on the same (timeline, state) — some recordings annotate
       the same event twice.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return events
        events = events.copy()

        is_seiz = events["type"] == "Seizure"
        if is_seiz.any():
            seiz = events.loc[is_seiz].drop_duplicates(
                subset=["timeline", "start", "duration", "state"]
            )
            events = pd.concat([events.loc[~is_seiz], seiz], ignore_index=True)

        events = events[
            ~((events["type"] == "Seizure") & (events["state"].isin(["bckg", "seiz"])))
        ].copy()

        events = events[~events["timeline"].isin(TUSZ_BAD_TIMELINES)].copy()

        events = events.sort_values(["timeline", "state", "start"]).reset_index(drop=True)
        if "stop" not in events.columns:
            events["stop"] = events["start"] + events["duration"]
        same = (
            (events["type"] == "Seizure")
            & (events["type"].shift() == "Seizure")
            & (events["timeline"] == events["timeline"].shift())
            & (events["state"] == events["state"].shift())
        )
        inter = np.minimum(events["stop"], events["stop"].shift()) - np.maximum(
            events["start"], events["start"].shift()
        )
        ratio = inter / np.minimum(events["duration"], events["duration"].shift())
        dup = same & (ratio > 0.99)
        events = events.loc[~dup].reset_index(drop=True)

        return events
