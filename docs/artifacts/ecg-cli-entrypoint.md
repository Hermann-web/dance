# ECG CLI Entrypoint

Date: 2026-05-28

Added command:

- `dance ecg-ludb-train --root <path> --records <id...> [--epochs N --batch-size N --lr LR --duration S --n-queries Q --device cpu|cuda]`

Behavior:

- Builds LUDB DataLoader through `dance.ecg.training.build_ludb_loader`.
- Instantiates `Dance(n_channels=1, n_classes=4, ...)`.
- Runs `train_one_epoch` for the requested number of epochs and prints loss.
