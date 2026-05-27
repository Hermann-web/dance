# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import numpy as np

from dance.training.extractors import EventEncoder


class _StubEvent:
    """Stand-in for a neuralset Event with the attributes our code reads."""

    def __init__(self, start, duration, *, description=None, state=None):
        self.start = start
        self.duration = duration
        self.description = description
        self.state = state


def _consume(extractor, events, start=0.0, duration=1.0):
    return next(extractor._get_timed_arrays(events, start, duration)).data


def test_dense_class_paints_class_labels_over_event_intervals():
    """dense_class mode paints each event's class index across its support
    in a single (1, T) mask consumed by the dense head."""
    extractor = EventEncoder(
        event_types="Stimulus",
        frequency=10.0,
        mode="dense_class",
        mapping={"Target": 1, "NonTarget": 2},
    )
    events = [
        _StubEvent(start=0.2, duration=0.3, description="Target"),
        _StubEvent(start=0.6, duration=0.2, description="NonTarget"),
    ]
    arr = _consume(extractor, events, start=0.0, duration=1.0)
    assert arr.shape == (1, 10)
    assert (arr[0, 2:5] == 1).all()  # Target spans samples 2..5
    assert (arr[0, 6:8] == 2).all()  # NonTarget spans samples 6..8


def test_event_class_packs_class_indices_into_first_column():
    """event_class mode packs up to max_events class indices into the
    first column of an (max_events, T) tensor for the DETR target loader."""
    extractor = EventEncoder(
        event_types="Stimulus",
        frequency=4.0,
        mode="event_class",
        max_events=3,
        mapping={"Target": 1, "NonTarget": 2},
    )
    events = [
        _StubEvent(start=0.1, duration=0.05, description="Target"),
        _StubEvent(start=0.4, duration=0.05, description="NonTarget"),
    ]
    arr = _consume(extractor, events, start=0.0, duration=1.0)
    np.testing.assert_array_equal(arr[:, 0], [1, 2, 0])
