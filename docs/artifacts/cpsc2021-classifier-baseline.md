# CPSC2021 Logistic-Regression Baseline

This artifact freezes the simple binary AF classifier path that complements the
event-detector path on `CPSC2021`.

## Purpose

The repository now supports two explicit `CPSC2021` regimes:

- interval detection through DANCE for AF episode localization;
- binary AF classification through a simple `scikit-learn` logistic-regression
  baseline.

The classifier exists because many Q1 `CPSC2021` papers evaluate fixed-window
or record-level AF detection with `accuracy`, `sensitivity`, `specificity`, and
`F1`, which is not the same task as interval localization.

## Module

`dance.ecg.classifier`

Key helpers:

- `build_cpsc2021_classification_table(...)`
- `extract_ecg_rr_features(...)`
- `evaluate_binary_classifier(...)`
- `run_cpsc2021_logreg_baseline(...)`

## Window Contract

- canonical reader: `dance.ecg.datasets.cpsc2021.read_cpsc2021_record`
- canonical labels: AF-positive if any dataset-native AF episode overlaps the
  window
- default window duration: `30.0` seconds
- default stride: `15.0` seconds
- subject identity: infer by dropping the final underscore suffix from the
  record stem when present
  - example: `data_31_1` and `data_31_2` belong to subject `data_31`

## Feature Contract

The baseline uses simple waveform and RR-derived features:

- signal mean, std, RMS, abs-mean
- signal IQR, peak-to-peak amplitude, line length, zero-crossing rate
- R-peak rate
- RR mean, RR std, RMSSD, pNN50
- median heart rate

This is intentionally simple. It is a comparison baseline, not the canonical
ECG model direction.

## CLI

```bash
uv run dance ecg-cpsc2021-logreg \
  --root /path/to/cpsc2021 \
  --train-records data_1_1 data_2_1 \
  --test-records data_31_1 data_32_1 \
  --lead 0 \
  --duration 30.0 \
  --stride 15.0
```

Subject-aware cross-validation:

```bash
uv run dance ecg-cpsc2021-logreg-cv \
  --root /path/to/cpsc2021 \
  --records data_1_1 data_1_2 data_2_1 data_2_2 data_3_1 data_3_2 \
  --lead 0 \
  --duration 30.0 \
  --stride 15.0 \
  --n-splits 5
```

## Reported Metrics

- `accuracy`
- `sensitivity`
- `specificity`
- `precision`
- `f1`
- `auroc`
- `average_precision`
