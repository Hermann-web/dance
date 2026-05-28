# CPSC2021 Data Contract

Date: 2026-05-28

Canonical reader:

- `dance.ecg.datasets.cpsc2021.read_cpsc2021_record`

Canonical ingestion:

- `WFDB` `rdrecord(...)`
- `WFDB` `rdann(..., extension="atr")`

Canonical rhythm interpretation:

- record-level rhythm class comes from header comments:
  - `non atrial fibrillation`
  - `persistent atrial fibrillation`
  - `paroxysmal atrial fibrillation`
- AF/AFL episode transitions come from `ann.aux_note`, not beat `symbol`
  values:
  - open markers: `(AFIB`, `(AFL`
  - close marker: `(N`

Canonical interval label:

- `af_episode`

Windowing:

- interval training and evaluation windows are built through the shared
  `_resolve_window_samples(...)` contract
- default standalone training/evaluation windows:
  - duration: `30.0` seconds
  - stride: `15.0` seconds

Subject grouping:

- subject identity is inferred from the record stem by dropping the final
  underscore suffix when present
  - example: `data_31_1` and `data_31_2` map to subject `data_31`

Supported benchmark regimes:

1. interval segmentation / episode detection
   - metrics:
     - `episode_f1`
     - `onset_delay`
     - `offset_delay`
     - `burden_error`
     - `mean_matched_iou`
   - optional official-style endpoint score through
     `dance.ecg.cpsc2021_score`

2. binary AF classification
   - labels are AF-positive if any canonical `af_episode` overlaps the window
   - default baseline:
     `dance.ecg.classifier.run_cpsc2021_logreg_baseline`
   - subject-aware CV:
     `dance.ecg.benchmark.run_cpsc2021_logreg_cv`
