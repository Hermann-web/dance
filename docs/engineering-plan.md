# Engineering Plan: ECG-First DANCE

## Context

DANCE’s model core is portable to ECG, but this repository is not ECG-ready at
the data and evaluation layers.

The current baseline is EEG-shaped in four concrete places:

- [dance/training/data.py](/home/ubuntu/Github/thesis/dance/dance/training/data.py:1)
  builds loaders from `neuralset` studies and expects a neuro extractor plus
  `EventEncoder` features.
- [dance/training/extractors.py](/home/ubuntu/Github/thesis/dance/dance/training/extractors.py:1)
  only knows EEG-oriented event families via `_LABEL_FIELD`:
  `Stimulus`, `Artifact`, and `Seizure`.
- [dance/metrics.py](/home/ubuntu/Github/thesis/dance/dance/metrics.py:1)
  scores interval events with IoU-based event F1, which is appropriate for span
  tasks but not for point-event ECG tasks.
- [dance/configs/defaults.yaml](/home/ubuntu/Github/thesis/dance/dance/configs/defaults.yaml:1)
  defaults to `EegExtractor`, EEG filtering, EEG channel handling, and EEG
  dataset configs.

The model contract itself is narrower and reusable:

- [dance/dance.py](/home/ubuntu/Github/thesis/dance/dance/dance.py:1)
  consumes a multichannel time series plus event spans and class ids.

The migration must therefore optimize for ECG-native data and metrics first,
not for preserving the current EEG plumbing.

## Product Thesis

This repo is worth continuing if it becomes a strong ECG event detector, not an
EEG codebase with ECG names painted over it.

The durable value is:

- reuse DANCE’s event-set detector where ECG truly is an event-localization
  problem;
- keep the model core when it helps;
- replace the data, labeling, and evaluation assumptions when ECG demands it.

## Anti-Thesis

This work must not become:

- an EEG compatibility exercise that keeps EEG semantics as the source of truth;
- a benchmark tour across many ECG datasets before one task works end to end;
- a heuristic-label pipeline that trains on `NeuroKit2`, `BioSPPy`, or
  `HeartPy` outputs instead of authoritative dataset annotations;
- a record-level diagnosis fork centered on PTB-XL or MIMIC-IV-ECG before any
  event-localization baseline exists;
- a neuralset integration project that delays the first ECG result.

## Quality Bar

The migration is only successful if all of the following are true:

- the first ECG path trains on dataset-native annotations with no EEG-only label
  assumptions;
- the public ECG code uses ECG semantics and quarantines legacy EEG naming to
  narrow compatibility adapters only;
- metrics match the actual ECG task rather than forcing ECG into inherited EEG
  scoring;
- another agent can implement the next phase without reopening task selection,
  dataset selection, ingestion tooling, or label representation strategy.

“Runs on ECG tensors” is not enough.

## Objective

Make DANCE an ECG-native event detector by shipping one correct first task and
one queued clinical follow-on path, while preserving the reusable model core and
avoiding premature rework around EEG-era data abstractions.

## Direction Chosen

The migration direction is frozen unless the user edits
[docs/open-decisions.md](/home/ubuntu/Github/thesis/dance/docs/open-decisions.md:1).

Primary path:

- first task: `LUDB` wave delineation;
- first problem formulation: interval events for `P`, `QRS`, and `T` waves using
  dataset-native delineation boundaries;
- first data layer: custom PyTorch ECG dataloader, not `neuralset`;
- first ingestion library: `WFDB`.

Secondary path, already in the pipeline after the first path is stable:

- second task: `CPSC2021` rhythm episode detection;
- second problem formulation: rhythm episode spans such as AF intervals;
- reuse the same ECG-native event schema and training surface, then add
  episode-specific metrics and sampling.

Deferred path:

- beat-level point-event tasks such as MIT-BIH-style beat detection or
  classification are deferred until interval-native LUDB and CPSC2021 baselines
  exist.

## Tooling Decision

Canonical ECG tooling:

- `WFDB` is the canonical reader and writer for the first migration path.
  Official docs expose multichannel record reading, channel selection, PhysioNet
  streaming, and annotation handling through `rdrecord`, `rdsamp`, and WFDB
  annotations.

Allowed but non-canonical helper tooling:

- `NeuroKit2` may be used for exploratory QA, visual sanity checks, or weak
  analysis utilities only. Its `ecg_process()` entrypoint is documented around a
  raw single-channel ECG pipeline and is not the training-label source of truth
  for multilead delineation datasets.

Not part of the planned core stack:

- `BioSPPy` is not a core dependency. Its ECG docs explicitly assume a
  single-channel Lead I-like signal, which is the wrong default for the LUDB
  first path.
- `HeartPy` is not a core dependency. Its main processing API is built around
  1D heart-rate signal processing and peak enhancement, which is useful for
  ad-hoc signal inspection but not as the canonical event dataset layer for this
  repo.

Dependency rule:

- add `scipy` explicitly to project dependencies before serious ECG work
  continues, because the repository already imports SciPy transitively.

## Hard Contracts

1. ECG semantics own the migration.
   Existing internal compatibility shims may still feed `batch["eeg"]` into the
   model, but no new public ECG-facing module, dataset wrapper, schema, or plan
   may treat EEG naming as canonical.

2. Dataset-native annotations are the training truth.
   Do not train the first ECG baselines on labels generated by `NeuroKit2`,
   `BioSPPy`, `HeartPy`, or hand-written QRS heuristics when annotated datasets
   already provide targets.

3. Event representation must be ECG-native, not EEG-inherited.
   LUDB uses wave spans. CPSC2021 uses rhythm episode spans. Point-event tasks
   stay deferred until interval baselines exist.

4. The first ECG baseline must not block on `neuralset`.
   If `neuralset` later gains an ECG-native extractor path, that is a follow-on
   integration task, not a prerequisite for the first result.

5. The model core may be preserved, but the metric layer is not sacred.
   Reuse the DANCE architecture where it fits. Replace task metrics and label
   rendering whenever ECG requires it.

6. `WFDB` is the canonical ingress for WFDB/PhysioNet-style datasets.
   Do not build bespoke parsers for LUDB, MIT-BIH, or CPSC2021 when WFDB
   already exposes the relevant signal and annotation primitives.

7. `LUDB` comes before `CPSC2021`, and `CPSC2021` comes before beat-level ECG.
   Agents must not reopen dataset ordering casually.

## Architecture Boundaries

Create an ECG-specific surface instead of mutating EEG abstractions until they
become meaningless.

New owned area:

- `dance/ecg/`
  - `datasets/`: dataset-native readers and manifests;
  - `events/`: ECG event schemas and label mappings;
  - `transforms/`: ECG-specific preprocessing and normalization;
  - `metrics/`: wave and rhythm metrics;
  - `data.py` or `dataloader.py`: custom ECG dataset + collate layer;
  - `adapters.py`: narrow compatibility layer into the existing `Dance` batch
    contract.

Existing modules that may be changed, but only for narrow compatibility:

- [dance/dance.py](/home/ubuntu/Github/thesis/dance/dance/dance.py:1):
  allow a compatibility alias if needed, but do not redesign the model around
  EEG assumptions.
- [pyproject.toml](/home/ubuntu/Github/thesis/dance/pyproject.toml:1):
  declare explicit ECG dependencies once they are actually part of the plan.
- [dance/cli](/home/ubuntu/Github/thesis/dance/dance/cli/main.py:1):
  integrate ECG datasets only after the standalone ECG path is stable.

Modules that must not be the first blocker:

- [dance/training/data.py](/home/ubuntu/Github/thesis/dance/dance/training/data.py:1)
- [dance/training/extractors.py](/home/ubuntu/Github/thesis/dance/dance/training/extractors.py:1)

These remain phase-two integration targets, not phase-one gates.

## In Scope

- define and freeze ECG-first task order and tooling;
- implement a custom ECG dataloader path for LUDB;
- define ECG-native event schemas for LUDB and CPSC2021;
- add ECG-appropriate metrics for wave delineation and rhythm episodes;
- preserve the reusable DANCE model core through a compatibility adapter;
- add explicit dependency declarations required by the chosen path;
- add durable artifacts that prevent future agents from reopening strategy.

## Out of Scope

- PTB-XL and MIMIC-IV-ECG as first-class migration targets;
- beat-level point-event training as the first or second milestone;
- channel-geometry research for a new ECG analogue of `ChannelMerger`;
- large-scale pretraining on Icentia11k before smaller baselines work;
- forcing ECG into `neuralset` before the custom dataloader path proves out.

## Durable Artifacts

- plan: [docs/engineering-plan.md](/home/ubuntu/Github/thesis/dance/docs/engineering-plan.md:1)
- decision override surface:
  [docs/open-decisions.md](/home/ubuntu/Github/thesis/dance/docs/open-decisions.md:1)
- implementation artifacts folder:
  [docs/artifacts](/home/ubuntu/Github/thesis/dance/docs/artifacts)

Expected artifacts to add during execution:

- `docs/artifacts/ecg-event-schema.md`
- `docs/artifacts/ludb-data-contract.md`
- `docs/artifacts/cpsc2021-data-contract.md`
- `docs/artifacts/windowing-calculations.md`

## Execution Order

### Phase 0: Freeze Strategy

- [ ] Keep this plan and `open-decisions` file aligned.
- [ ] Do not start coding from a different first dataset or task family unless
      `open-decisions` changes.

Done when:

- the first dataset, first task, ingress tooling, and fallback policy are frozen
  in durable docs.

### Phase 1: LUDB ECG Baseline

- [x] Create `dance/ecg/` as the owned ECG implementation surface.
- [x] Implement LUDB loading with `WFDB`.
- [x] Convert LUDB wave delineation annotations into ECG-native interval events.
- [x] Define a collate path that feeds the existing `Dance` model through a
      narrow adapter rather than through `neuralset`.
- [x] Implement wave-delineation metrics:
      onset MAE, offset MAE, tolerance F1, and interval event F1.
- [x] Add tests for the LUDB reader, event conversion, batch contract, and
      metrics.

Done when:

- one LUDB batch can be read, collated, and consumed by the model;
- tests prove event spans are built correctly from source annotations;
- ECG-facing code no longer depends on EEG label semantics.

### Phase 2: Training Surface and Hygiene

- [x] Add explicit dependencies required by the ECG path.
- [x] Decide whether `Dance` needs a public `signal` alias or whether the ECG
      adapter should remain the only compatibility boundary.
- [x] Add a minimal training entrypoint for the standalone ECG path.
- [x] Integrate ECG into the CLI/config system only after the standalone path
      is stable.

Done when:

- LUDB can be trained through a repeatable repository-native command path;
- the compatibility boundary between ECG modules and legacy EEG naming is
  explicit and tested.

### Phase 3: CPSC2021 Clinical Follow-On

- [x] Add `CPSC2021` ingestion via `WFDB` or dataset-native wrappers that still
      terminate in the same ECG event schema.
- [x] Implement rhythm episode preprocessing and event merging rules.
- [x] Add episode metrics:
      episode F1, onset delay, offset delay, and burden error.
- [x] Add sampling and class-imbalance handling appropriate for rare rhythm
      episodes.

Done when:

- the same ECG training surface can run both LUDB and CPSC2021 with task-fit
  metrics and no EEG-specific assumptions.

### Phase 4: Optional NeuralSet Integration

- [ ] Re-evaluate `neuralset` only after LUDB and CPSC2021 are stable.
- [ ] If integration is still worthwhile, add an ECG-native extractor or study
      bridge that preserves the already-proven ECG contracts.

Done when:

- `neuralset` support reduces maintenance cost rather than blocking progress or
  reintroducing EEG-centric abstractions.

## Verification Model

Every phase must be backed by repo-local verification, not only prose.

Required verification for Phase 1:

- unit tests for ECG event conversion;
- unit tests for collate and adapter behavior;
- unit tests for wave delineation metrics;
- one smoke test that executes a forward pass of `Dance` on an ECG batch.

Required verification for Phase 3:

- unit tests for rhythm episode conversion and merging;
- unit tests for rhythm metrics;
- one smoke test on a CPSC2021-shaped batch.

Known current protection:

- [dance/tests/test_data.py](/home/ubuntu/Github/thesis/dance/dance/tests/test_data.py:1)
  already protects the DETR collate rename pattern.
- [dance/tests/test_extractors.py](/home/ubuntu/Github/thesis/dance/dance/tests/test_extractors.py:1)
  already protects the current event encoder behavior.

Known current gap:

- there are no ECG-specific tests, no ECG dataset readers, no ECG metrics, and
  no test that verifies ECG-native labels through the model path.

## Progress Notes

2026-05-28

- Strategy frozen: `LUDB` first, `CPSC2021` second.
- Ingestion frozen: `WFDB` is the canonical ECG reader.
- Label policy frozen: use dataset-native ECG annotations; skip heuristic label
  generation for core training.
- Data-layer decision frozen: custom PyTorch ECG dataloader first, `neuralset`
  later if still useful.
- Implemented `dance/ecg/` Phase 1 baseline surface with LUDB WFDB reader,
  ECG-native wave schema, custom dataset/collate, compatibility adapter, and
  ECG metric wrapper + event conversion helper.
- Added `docs/artifacts/ecg-event-schema.md` and
  `docs/artifacts/ludb-data-contract.md` to lock contracts.
- Added standalone ECG training bootstrap in `dance.ecg.training` plus
  `docs/artifacts/ecg-training-entrypoint.md`.
- Added `Dance.forward` compatibility for `signal` alias (mapped to canonical
  internal `eeg`) while retaining ECG adapter as primary boundary.
- Added standalone CLI command `dance ecg-ludb-train` for repository-native
  LUDB training runs via ECG modules.
- Added initial CPSC2021 WFDB reader + AF episode conversion/merge and initial
  rhythm metric surface (episode F1, onset delay, offset delay, burden error).
- Added initial CPSC2021 dataset/collate and weighted sampler helper for
  AF-episode class imbalance handling.
- Added `build_cpsc2021_loader` and CLI command `dance ecg-cpsc2021-train` so
  the same standalone ECG training surface now runs both LUDB and CPSC2021.
