# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import argparse
import sys

from .. import __version__
from .helpers import build_config, list_datasets


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dance",
        description="DANCE: Detect and Classify Events in EEG.",
    )
    parser.add_argument("--version", action="version", version=f"dance {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-datasets", help="Print the available dataset slugs.")

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
