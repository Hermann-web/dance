# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import argparse
import sys

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from .. import __version__
from .helpers import build_config, list_datasets


def _parse_lead(value: str) -> str | int:
    """Parse CLI lead argument as int index or lead-name string."""
    raw = value.strip()
    if raw == "":
        raise ValueError("Lead cannot be empty.")
    if "." in raw:
        raise ValueError(
            f"Lead {raw!r} is not valid: use an integer index (e.g. 0, -1) or lead name."
        )
    try:
        return int(raw)
    except ValueError:
        return raw


class _EcgCliTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: PositiveInt
    epochs: PositiveInt
    n_queries: PositiveInt
    n_classes: PositiveInt
    lr: PositiveFloat
    duration: PositiveFloat


def _run_ecg_train(
    *,
    root: str,
    records: list[str],
    lead: str | int,
    batch_size: int,
    lr: float,
    epochs: int,
    duration: float,
    n_queries: int,
    device: str,
    n_classes: int,
    build_loader,
) -> int:
    import torch

    from ..dance import Dance
    from ..ecg.training import train_one_epoch

    _EcgCliTrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        n_queries=n_queries,
        n_classes=n_classes,
        lr=lr,
        duration=duration,
    )
    if not records:
        raise ValueError("records must contain at least one record id.")
    loader = build_loader(
        root=root,
        record_ids=records,
        lead=lead,
        batch_size=batch_size,
        shuffle=True,
    )
    model = Dance(
        n_channels=1,
        n_classes=n_classes,
        n_queries=n_queries,
        duration=duration,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optim, device=device)
        print(f"epoch={epoch + 1} loss={loss:.6f}")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dance",
        description="DANCE: Detect and Classify Events in EEG.",
    )
    parser.add_argument("--version", action="version", version=f"dance {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-datasets", help="Print the available dataset slugs.")
    ecg = sub.add_parser("ecg-ludb-train", help="Run minimal standalone ECG LUDB training.")
    ecg.add_argument("--root", required=True, help="LUDB root folder with WFDB record files.")
    ecg.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="Record stems (e.g. 1 2 3...) to train on.",
    )
    ecg.add_argument("--batch-size", type=int, default=8)
    ecg.add_argument("--lead", type=str, default="0", help="Lead index or lead name.")
    ecg.add_argument("--lr", type=float, default=1e-3)
    ecg.add_argument("--epochs", type=int, default=1)
    ecg.add_argument("--duration", type=float, default=1.0)
    ecg.add_argument("--n-queries", type=int, default=64)
    ecg.add_argument("--device", type=str, default="cpu")
    ecg_rhythm = sub.add_parser(
        "ecg-cpsc2021-train",
        help="Run minimal standalone ECG CPSC2021 rhythm training.",
    )
    ecg_rhythm.add_argument("--root", required=True, help="CPSC2021 root folder.")
    ecg_rhythm.add_argument("--records", nargs="+", required=True, help="Record stems.")
    ecg_rhythm.add_argument("--batch-size", type=int, default=8)
    ecg_rhythm.add_argument("--lead", type=str, default="0", help="Lead index or lead name.")
    ecg_rhythm.add_argument("--lr", type=float, default=1e-3)
    ecg_rhythm.add_argument("--epochs", type=int, default=1)
    ecg_rhythm.add_argument("--duration", type=float, default=1.0)
    ecg_rhythm.add_argument("--n-queries", type=int, default=64)
    ecg_rhythm.add_argument("--device", type=str, default="cpu")

    run = sub.add_parser("run", help="Train + test DANCE on one dataset.")
    run.add_argument("dataset", help="Dataset slug (see `dance list-datasets`).")

    # Running mode: at most one of --debug / --local / --submit. The
    # default (no flag) is the same as --local but with a single config
    # (no grid sweep) — i.e. plain "run the YAML as-is".
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--debug",
        action="store_true",
        help="1 epoch, 1 GPU, no W&B. Quick sanity check.",
    )
    mode.add_argument(
        "--local",
        action="store_true",
        help="Run the full reproduction grid sequentially in-process "
        "(no SLURM needed). Slow but no infrastructure required.",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit the full reproduction grid to SLURM. Recommended.",
    )

    # Shared overrides (apply to all modes)
    run.add_argument("--epochs", type=int, default=None, help="Override n_epochs.")
    run.add_argument("--gpus", type=int, default=None, help="Override gpus_per_node.")
    run.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    run.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Override infra.folder (where results / checkpoints land).",
    )
    run.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="exca cache root. Defaults to $XDG_CACHE_HOME/dance.",
    )
    run.add_argument(
        "--study-path",
        type=str,
        default=None,
        help="Where the raw study lives. MOABB datasets auto-download here "
        "on first use; TUH datasets (shah2018) require an existing corpus.",
    )

    # Grid-only options (only effective with --submit / --local)
    grp = run.add_argument_group("grid options (used with --submit / --local)")
    grp.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to sweep over. Default: [33] for non-TUSZ, "
        "[33, 34, 35] for TUSZ.",
    )
    grp.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="K-fold indices to sweep over (ignored for TUSZ). Default: 0..4.",
    )
    grp.add_argument(
        "--grid-name",
        type=str,
        default=None,
        help="Grid name (used as W&B group and output sub-folder). "
        "Default: e.g. korczowski2014a -> BI2014A.",
    )
    grp.add_argument("--project", default="DANCE", help="W&B project. Default: DANCE.")
    grp.add_argument("--wandb-host", default=None, help="W&B host. Default: wandb.com.")
    grp.add_argument(
        "--mode",
        choices=("cached", "retry", "force"),
        default="retry",
        help="exca infra_mode (default: retry).",
    )
    grp.add_argument(
        "--slurm-partition",
        default=None,
        help="SLURM partition (--submit only). Default: cluster-defined.",
    )
    grp.add_argument(
        "--timeout-min",
        type=int,
        default=72 * 60,
        help="SLURM job timeout in minutes (--submit only). Default: 72 h.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.cmd == "list-datasets":
        print("\n".join(list_datasets()))
        return 0
    if args.cmd == "ecg-ludb-train":
        from ..ecg.events import WAVE_CLASS_TO_ID
        from ..ecg.training import build_ludb_loader

        return _run_ecg_train(
            root=args.root,
            records=args.records,
            lead=_parse_lead(args.lead),
            batch_size=args.batch_size,
            duration=args.duration,
            n_queries=args.n_queries,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device,
            n_classes=len(WAVE_CLASS_TO_ID),
            build_loader=build_ludb_loader,
        )
    if args.cmd == "ecg-cpsc2021-train":
        from ..ecg.events import RHYTHM_CLASS_TO_ID
        from ..ecg.training import build_cpsc2021_loader

        return _run_ecg_train(
            root=args.root,
            records=args.records,
            lead=_parse_lead(args.lead),
            batch_size=args.batch_size,
            duration=args.duration,
            n_queries=args.n_queries,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device,
            n_classes=len(RHYTHM_CLASS_TO_ID),
            build_loader=build_cpsc2021_loader,
        )

    if args.cmd == "run":
        if args.submit or args.local:
            from ..training.grid import run_grid_from_args

            return run_grid_from_args(args)

        # Plain `dance run NAME` (with or without --debug): single config.
        from ..training.main import Experiment

        cfg = build_config(
            args.dataset,
            debug=args.debug,
            epochs=args.epochs,
            gpus=args.gpus,
            seed=args.seed,
            folder=args.folder,
            cache_folder=args.cache_folder,
            study_path=args.study_path,
        )
        exp = Experiment(**cfg)
        exp.infra.clear_job()
        exp.run()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
