# ECG End-to-End Training Guide (LUDB + CPSC2021)

Date: 2026-05-28

This guide explains:

1. how to collect ECG data,
2. how ingestion works in this repo,
3. what processing happens before training,
4. how to run end-to-end training,
5. how to run the simple classifier baseline,
6. how to run subject-wise classifier CV and challenge scoring,
7. how to read results.

---

## 1) Prerequisites

- Python 3.12+
- Install package and dependencies:

```bash
uv sync
```

The ECG pipeline relies on:

- `wfdb` for dataset ingestion
- `scipy` and `numpy` for signal/data operations
- `check-shapes` + `pydantic` for batch/CLI validation

---

## 2) Data collection

Use WFDB-compatible copies of datasets locally.

- LUDB root directory should contain record files by stem (e.g. `1.hea`, `1.dat`, `1.atr`).
- CPSC2021 root directory should contain record files by stem (e.g. `A001.hea`, `A001.dat`, `A001.atr`).

You can use PhysioNet tooling or WFDB APIs externally to download those records.

---

## 3) Ingestion + preprocessing in this repo

### LUDB path

- Reader: `dance.ecg.datasets.ludb.read_ludb_record`
- Annotations are converted to interval events (`p_wave`, `qrs_complex`, `t_wave`).
- Dataset wrapper: `dance.ecg.data.LudbDataset`
- Collate: `dance.ecg.data.ludb_collate` (via shared interval collator)

### CPSC2021 path

- Reader: `dance.ecg.datasets.cpsc2021.read_cpsc2021_record`
- AF episode markers are converted to interval episodes (`af_episode`) with optional merge.
- Dataset wrapper: `dance.ecg.rhythm_data.Cpsc2021Dataset`
- Collate: `dance.ecg.rhythm_data.cpsc2021_collate` (via shared interval collator)

### Common processing

- ECG records are segmented into explicit training windows before collation.
- Raw sample boundaries are normalized to `[0, 1]` by the active window length
  in collate.
- Batch adapter `ecg_batch_to_dance_batch` validates:
  - required keys
  - shape consistency
  - normalized boundaries
  - class dtype
  - optional channel-position shape
- The standalone ECG CLI trains with `use_channel_merger=False` and does not
  synthesize fake channel geometry.
- `Dance.forward` accepts canonical `eeg`; also supports `signal` alias.

---

## 4) End-to-end training commands

### LUDB

```bash
uv run dance ecg-ludb-train \
  --root /path/to/ludb \
  --records 1 2 3 4 5 \
  --lead 0 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-3 \
  --duration 4.0 \
  --stride 2.0 \
  --n-queries 64 \
  --checkpoint-out results/ecg/ludb/model.ckpt \
  --device cpu
```

### CPSC2021

```bash
uv run dance ecg-cpsc2021-train \
  --root /path/to/cpsc2021 \
  --records data_31_1 data_31_2 data_31_3 \
  --lead 0 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-3 \
  --duration 30.0 \
  --stride 15.0 \
  --n-queries 64 \
  --checkpoint-out results/ecg/cpsc2021/model.ckpt \
  --device cpu
```

---

## 5) Simple CPSC2021 classifier baseline

```bash
uv run dance ecg-cpsc2021-logreg \
  --root /path/to/cpsc2021 \
  --train-records data_1_1 data_2_1 data_3_1 \
  --test-records data_31_1 data_32_1 \
  --lead 0 \
  --duration 30.0 \
  --stride 15.0
```

The classifier path is intentionally separate from DANCE interval detection. It
matches the common Q1 `CPSC2021` binary AF evaluation regime more directly than
the event detector does.

---

## 6) Subject-wise CV and challenge scoring

Subject-aware classifier CV:

```bash
uv run dance ecg-cpsc2021-logreg-cv \
  --root /path/to/cpsc2021 \
  --records data_1_1 data_1_2 data_2_1 data_2_2 data_3_1 data_3_2 \
  --lead 0 \
  --duration 30.0 \
  --stride 15.0 \
  --n-splits 5
```

Official-style endpoint scoring from a JSON predictions map:

```bash
uv run dance ecg-cpsc2021-score \
  --root /path/to/cpsc2021 \
  --predictions-json /path/to/predictions.json
```

Prediction JSON format:

```json
{
  "data_31_1": [[1200, 5600]],
  "data_31_2": []
}
```

---

## 7) Result reading

Held-out detector evaluation:

```bash
uv run dance ecg-ludb-eval \
  --root /path/to/ludb \
  --records 6 7 8 \
  --lead 0 \
  --duration 4.0 \
  --stride 2.0 \
  --checkpoint results/ecg/ludb/model.ckpt
```

```bash
uv run dance ecg-cpsc2021-eval \
  --root /path/to/cpsc2021 \
  --records data_41_1 data_42_1 \
  --lead 0 \
  --duration 30.0 \
  --stride 15.0 \
  --checkpoint results/ecg/cpsc2021/model.ckpt
```

CLI training prints:

```text
epoch=1 loss=...
epoch=2 loss=...
...
```

Interpretation:

- loss is the mean batch loss for that epoch from `train_one_epoch`.
- compare trend across epochs (decreasing is typically expected in stable runs).
- for detector-quality metrics, use the helpers in `dance.ecg.evaluation`.
- for CPSC2021, weighted sampling is enabled in the standalone CLI to reduce
  under-sampling of AF-positive windows.
- the detector eval CLIs print JSON metric reports for the held-out record set.
- the logistic-regression CLI prints JSON with `accuracy`, `sensitivity`,
  `specificity`, `precision`, `f1`, `auroc`, and dataset window counts.
- the CV CLI prints per-fold metrics plus aggregate mean/std across folds.
- the challenge scorer prints per-record scores and the mean official-style
  score.

---

## 8) Reproducible bash automation

Use the included scripts:

- `scripts/ecg/train_ludb_e2e.sh`
- `scripts/ecg/train_cpsc2021_e2e.sh`

They perform:

1. environment check (`dance` CLI availability),
2. dataset file checks for each requested record,
3. training run,
4. tee logs to timestamped output under `results/ecg/<dataset>/`.
