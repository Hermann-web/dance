# ECG Event Schema (Phase 1)

Date: 2026-05-28

## Canonical LUDB wave schema

- Event type: interval wave event with integer sample boundaries.
- Dataclass: `EcgWaveEvent(label, onset, offset)` in `dance/ecg/events/schema.py`.
- Class mapping:
  - `bg -> 0`
  - `p_wave -> 1`
  - `qrs_complex -> 2`
  - `t_wave -> 3`
- Annotation symbol mapping (WFDB `atr`):
  - `p -> p_wave`
  - `N -> qrs_complex`
  - `t -> t_wave`

## Conversion contract

- Input: WFDB annotation streams from LUDB `(sample, symbol)`.
- Event boundaries are constructed by matched open/close symbols:
  - open token: `(<wave-symbol>`
  - close token: `)<wave-symbol>`
- Unknown symbols are ignored.
- Only intervals with `offset > onset` are emitted.

## Batch contract at ECG boundary

- ECG collate emits normalized event tensors:
  - `start`: `(B, E)` in `[0, 1]`
  - `end`: `(B, E)` in `[0, 1]`
  - `class`: `(B, E)` integer ids with zero padding
- Compatibility adapter maps directly into the existing `Dance` inputs:
  - `eeg`, `start`, `end`, `class`
