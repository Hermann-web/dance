# ECG Training Entrypoint (Phase 2 bootstrap)

Date: 2026-05-28

- Module: `dance.ecg.training`
- Functions:
  - `build_ludb_loader(root, record_ids, duration, stride, batch_size, shuffle)`
  - `train_one_epoch(model, loader, optimizer, device="cpu")`

This is a minimal standalone ECG training surface that does not depend on
`neuralset` and uses LUDB contracts from `dance.ecg.data` and
`dance.ecg.adapters`.

The LUDB loader now windows each 10-second record into explicit training
segments, so `duration` is a real loader/window parameter rather than only a
model hyperparameter.
