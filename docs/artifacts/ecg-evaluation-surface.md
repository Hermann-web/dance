# ECG Evaluation Surface

This artifact locks the repository-native ECG evaluation contracts added after
the Q1 literature review.

## LUDB

Module: `dance.ecg.evaluation`

Primary helper: `evaluate_wave_events(...)`

Metric surface:

- event-level `event_f1`
- event-level `onset_mae`
- event-level `offset_mae`
- event-level `tolerance_f1`
- samplewise `sample_accuracy`
- samplewise `sample_macro_precision`
- samplewise `sample_macro_sensitivity`
- samplewise `sample_macro_specificity`
- samplewise `sample_macro_f1`

Rationale:

- event metrics match the DANCE detector formulation;
- samplewise metrics match the way many Q1 LUDB papers report segmentation
  quality for `P`, `QRS`, and `T`.

## CPSC2021 Interval Segmentation

Module: `dance.ecg.evaluation`

Primary helper: `evaluate_rhythm_events(...)`

Metric surface:

- `episode_f1`
- `onset_delay`
- `offset_delay`
- `burden_error`
- `mean_matched_iou`

Rationale:

- `episode_f1` and boundary delays preserve detector-native interval scoring;
- `mean_matched_iou` provides a cleaner bridge to interval-segmentation papers
  that report overlap quality more directly than event F1 alone.

Official challenge-style scoring:

- module: `dance.ecg.cpsc2021_score`
- CLI: `uv run dance ecg-cpsc2021-score --root ... --predictions-json ...`
- purpose: score sample-index AF endpoints with the published rhythm-plus-
  endpoint logic used by the official `CPSC2021` scorer

## Batch-Level Adapters

Helpers:

- `evaluate_wave_batch(...)`
- `evaluate_rhythm_batch(...)`

These decode DANCE DETR outputs through the existing batch contract and then
delegate to the task-level helpers above.
