from __future__ import annotations

from dataclasses import dataclass

WAVE_CLASS_TO_ID = {"bg": 0, "p_wave": 1, "qrs_complex": 2, "t_wave": 3}
WAVE_ANN_TO_CLASS = {"p": "p_wave", "N": "qrs_complex", "t": "t_wave"}


@dataclass(frozen=True, slots=True)
class EcgWaveEvent:
    """Canonical ECG wave delineation event in sample indices."""

    label: str
    onset: int
    offset: int

    @property
    def class_id(self) -> int:
        return WAVE_CLASS_TO_ID[self.label]
