# LUDB Data Contract (Phase 1)

Date: 2026-05-28

## Reader

- Canonical ingestion library: `wfdb`.
- Reader entrypoint: `dance.ecg.datasets.ludb.read_ludb_record(record_path, lead=...)`.
- Output:
  - `record_id: str`
  - `signal: np.ndarray[float32]` (single lead)
  - `fs: float`
  - `events: list[EcgWaveEvent]`

## Dataset

- Dataset class: `dance.ecg.data.LudbDataset`.
- Inputs:
  - `root`: LUDB record directory
  - `record_ids`: list of WFDB record stems
  - `lead`: lead name or integer index
- Item output:
  - `eeg`: `torch.Tensor[1, T]`
  - `event_start`: onset sample indices (float array)
  - `event_end`: offset sample indices (float array)
  - `event_class`: wave ids (int array)

## Collate

- Collate function: `dance.ecg.data.ludb_collate`.
- Pads signal length to longest record in batch.
- Pads event dimension to max events in batch.
- Converts event sample indices into normalized window positions:
  - `start = event_start / T`
  - `end = event_end / T`

## Adapter

- Adapter: `dance.ecg.adapters.ecg_batch_to_dance_batch`.
- Required keys:
  - `eeg`, `start`, `end`, `class`
