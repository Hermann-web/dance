# CPSC2021 Sampling (Initial)

Date: 2026-05-28

- Module: `dance.ecg.rhythm_data`
- Sampler helper: `build_rhythm_weighted_sampler(dataset, positive_weight, negative_weight)`
- Strategy: upweight records containing AF episodes to mitigate class imbalance
  in rhythm episode training.

Current scope:

- Record-level balancing by presence/absence of at least one AF episode in the
  windowed training item.
