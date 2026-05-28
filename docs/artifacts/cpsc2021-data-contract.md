# CPSC2021 Data Contract (Initial)

Date: 2026-05-28

- Reader: `dance.ecg.datasets.cpsc2021.read_cpsc2021_record`
- Canonical ingestion: WFDB (`rdrecord`, `rdann`)
- Episode conversion: `episodes_from_wfdb_ann`
  - open symbol: `(AFIB`
  - close symbol: `)AFIB`
  - label: `af_episode`
  - optional merge by `merge_gap_seconds`

Rhythm metric surface (initial):

- `EcgRhythmEpisodeF1`
- `EcgOnsetDelay`
- `EcgOffsetDelay`
- `EcgBurdenError`
