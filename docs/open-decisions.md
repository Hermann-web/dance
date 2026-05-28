# Open Decisions

This file is the durable override surface for the ECG migration.

Rule:

- Future agents must follow the latest contents of this file and
  [docs/engineering-plan.md](/home/ubuntu/Github/thesis/dance/docs/engineering-plan.md:1).
- If this file changes, its updated decisions override earlier frozen choices in
  the plan.
- If this file does not change, agents must not reopen the frozen decisions
  below.

## Frozen Decisions

### D1

- status: frozen
- decision: the first ECG task is `LUDB` wave delineation
- why: it is the cleanest event-localization fit and the fastest path to a
  correct ECG-native baseline

### D2

- status: frozen
- decision: the second ECG task is `CPSC2021` rhythm episode detection
- why: it is the stronger clinical follow-on path and keeps DANCE in an
  interval-native regime

### D3

- status: frozen
- decision: `WFDB` is the canonical ingestion library for the first ECG path
- why: it already handles WFDB records, multichannel reads, channel selection,
  PhysioNet access, and annotations

### D4

- status: frozen
- decision: dataset-native annotations are the only canonical training labels
- why: ECG migration must not be anchored to heuristic labels from helper
  libraries when authoritative annotations exist

### D5

- status: frozen
- decision: the first ECG baseline uses a custom PyTorch dataloader, not
  `neuralset`
- why: the repo’s current `neuralset` path is EEG-shaped and should not block
  the first ECG result

### D6

- status: frozen
- decision: `NeuroKit2` is optional QA tooling only; `BioSPPy` and `HeartPy`
  are not core dependencies
- why: they are useful for inspection and heuristics, but they are not the
  right canonical data layer for multilead annotated ECG event datasets

### D7

- status: frozen
- decision: beat-level point-event tasks are deferred until LUDB and CPSC2021
  baselines are stable
- why: the first two milestones should stay interval-native and metric-clean

## Change Log

2026-05-28

- created and aligned with the initial ECG engineering plan
