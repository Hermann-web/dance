# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import argparse
from pathlib import Path

from ..cli.helpers import build_config


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Register the `warm-cache` subcommand on the `dance` CLI."""
    p = sub.add_parser(
        "warm-cache",
        help="Pre-populate the exca caches for one dataset.",
        description="Pre-populate the study + neuro exca caches for one dataset.",
    )
    p.add_argument("dataset", help="Dataset slug.")
    p.add_argument(
        "--cache-folder",
        default=None,
        help="exca cache root. Default: $XDG_CACHE_HOME/dance.",
    )
    p.add_argument(
        "--study-path",
        default=None,
        help="Where the raw study lives (TUH datasets require this).",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit as a SLURM job (recommended for large datasets like TUSZ).",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=80,
        help="CPUs (== processpool workers) when --submit.",
    )
    p.add_argument(
        "--slurm-partition",
        default=None,
        help="SLURM partition (only used with --submit).",
    )
    p.add_argument(
        "--timeout-min",
        type=int,
        default=12 * 60,
        help="SLURM walltime when --submit.",
    )
    p.set_defaults(func=run)


def _warm(dataset: str, cache_folder: str | None, study_path: str | None) -> None:
    import neuralset as ns
    from exca import ConfDict

    from . import main as _main_module  # noqa: F401  registers discriminators
    from .data import Data

    cfg = build_config(
        dataset,
        debug=False,
        epochs=None,
        gpus=None,
        seed=None,
        folder=None,
        cache_folder=cache_folder,
        study_path=study_path,
    )
    data = Data(**ConfDict(cfg)["data"])
    try:
        data.study.download()
    except NotImplementedError:
        pass
    print(f"[warm-cache] {dataset}: study.run() ...", flush=True)
    events = data.study.run()
    print(
        f"[warm-cache] {dataset}: study cache populated ({len(events)} events)",
        flush=True,
    )
    if data.preprocessor is not None:
        events = data.preprocessor.run(events)
    events = ns.events.standardize_events(events)
    print(f"[warm-cache] {dataset}: neuro.prepare() ...", flush=True)
    data.neuro.prepare(events)
    print(f"[warm-cache] {dataset}: done", flush=True)


def run(args: argparse.Namespace) -> int:
    if not args.submit:
        _warm(args.dataset, args.cache_folder, args.study_path)
        return 0

    import submitit

    log_root = (
        Path(args.cache_folder) if args.cache_folder else _default_cache()
    ) / "_warmup_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_root / args.dataset))
    slurm_params: dict = {
        "slurm_constraint": "volta32gb",
        "cpus_per_task": args.cpus,
        "gpus_per_node": 0,
        "timeout_min": args.timeout_min,
        "slurm_job_name": f"warm_{args.dataset}",
    }
    if args.slurm_partition:
        slurm_params["slurm_partition"] = args.slurm_partition
    executor.update_parameters(**slurm_params)
    job = executor.submit(_warm, args.dataset, args.cache_folder, args.study_path)
    print(f"[warm-cache] {args.dataset}: submitted SLURM job {job.job_id}")
    print(f"             logs: {log_root / args.dataset}")
    return 0


def _default_cache() -> Path:
    from ..cli.helpers import default_cache_dir

    return default_cache_dir()
