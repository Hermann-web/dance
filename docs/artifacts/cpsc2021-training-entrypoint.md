# CPSC2021 Training Entrypoint

Date: 2026-05-28

- Loader helper: `dance.ecg.training.build_cpsc2021_loader`
- CLI command:
  - `dance ecg-cpsc2021-train --root <path> --records <id...> [--epochs N --batch-size N --lr LR --duration S --stride S --n-queries Q --device cpu|cuda]`

Uses the same training loop contract as LUDB:

- model: `Dance`
- loop: `train_one_epoch`
- batch adapter: `ecg_batch_to_dance_batch`

The repaired CPSC2021 path:

- reads AF rhythm transitions from WFDB `aux_note`;
- uses header comments to distinguish non-AF, persistent AF, and paroxysmal AF;
- windows long records before collation;
- wires weighted sampling into the standalone training command.
