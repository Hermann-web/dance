# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import typing as tp

import neuralset as ns
import numpy as np
from neuralset.extractors import BaseExtractor

_LABEL_FIELD: dict[str, str] = {
    "Stimulus": "description",
    "Artifact": "state",
    "Seizure": "state",
}


class EventEncoder(BaseExtractor):
    """Encode events into per-segment dense or sparse tensors.

    Parameters
    ----------
    event_types
        Underlying neuralset `Event` subclass name (e.g. `"Stimulus"`,
        `"Artifact"`, `"Seizure"`).
    frequency
        Output sampling rate in Hz. For `dense_class`, this is the
        model's dense output rate; for the `event_*` modes, this is
        the post-Perceiver token rate.
    mode
        Output layout: `dense_class` writes a per-timestep class id;
        `event_start` / `event_end` write window-relative timestamps;
        `event_class` writes the class id of each selected event.
    max_events
        Required for the `event_*` modes. Sets the first dimension of
        the output tensor (zero-padded if the window contains fewer
        events than `max_events`).
    mapping
        Categorical label to integer class index. Required for
        `dense_class` and `event_class`.
    """

    event_types: str = "Event"
    frequency: float = 100.0
    mode: tp.Literal[
        "dense_class",
        "event_start",
        "event_end",
        "event_class",
    ] = "dense_class"
    max_events: int | None = None
    mapping: dict[str, int] | None = None

    def _label(self, event) -> int | None:
        """Return the integer class for `event`, or `None` if its label
        is not in the mapping. Unknown labels are silently skipped so
        that one stray annotation (e.g. TUSZ's generic `"seiz"` when
        the mapping only knows the 8 subtype codes) cannot crash a long
        training run; add the label to the dataset YAML to include it.
        """
        if self.mapping is None:
            raise ValueError("mapping is required to look up event class indices")
        field = _LABEL_FIELD.get(self.event_types)
        if field is None:
            raise ValueError(
                f"Unsupported event_types={self.event_types!r}. "
                f"Supported: {sorted(_LABEL_FIELD)}"
            )
        return self.mapping.get(getattr(event, field))

    def _get_timed_arrays(self, events, start, duration):
        # BaseExtractor.prepare() calls us once with duration=0.001 to
        # learn the output shape; clamp so we never return a zero-width
        # time axis.
        n_samples = max(1, int(duration * self.frequency))

        if self.mode == "dense_class":
            data = np.zeros((1, n_samples), dtype=np.float32)
            for e in events:
                label = self._label(e)
                if label is None:
                    continue
                e_start = int((e.start - start) * self.frequency)
                e_end = int((e.start + e.duration - start) * self.frequency)
                if not (0 <= (e_start + e_end) // 2 < n_samples):
                    continue
                data[0, max(0, e_start) : min(n_samples, e_end)] = label
            yield ns.base.TimedArray(
                data=data, start=start, duration=duration, frequency=self.frequency
            )
            return

        if self.max_events is None:
            raise ValueError(f"max_events is required for mode={self.mode!r}")
        selected = [
            e
            for e in events
            if start <= e.start <= (e.start + e.duration) <= start + duration
        ]
        # When this feature carries a mapping (event_class / dense_target),
        # also drop events with labels outside the mapping. Features that
        # use timestamps only (event_start, event_end) have no mapping and
        # keep all windowed events.
        if self.mapping is not None:
            selected = [e for e in selected if self._label(e) is not None]
        if self.mode == "event_start":
            values = [(e.start - start) / duration for e in selected]
        elif self.mode == "event_end":
            values = [(e.start + e.duration - start) / duration for e in selected]
        elif self.mode == "event_class":
            values = [self._label(e) for e in selected]
        else:  # pragma: no cover - exhausted by Literal
            raise NotImplementedError(self.mode)

        values = np.pad(
            values[: self.max_events],
            (0, max(0, self.max_events - len(values))),
        )
        data = np.zeros((self.max_events, n_samples), dtype=np.float32)
        data[:, 0] = values
        yield ns.base.TimedArray(
            data=data, start=start, duration=duration, frequency=self.frequency
        )
