# ECG Compatibility Boundary Decision

Date: 2026-05-28

Decision:

- Keep `dance.ecg.adapters.ecg_batch_to_dance_batch` as the primary ECG
  compatibility boundary.
- Also allow a narrow public `signal` alias in `Dance.forward` so callers can
  pass ECG batches with `signal` and avoid forcing EEG-named keys at callsites.

Rationale:

- Preserves internal model assumptions with minimal risk (`eeg` remains
  canonical inside model code).
- Reduces friction for ECG-native integrations and future dataset wrappers.
