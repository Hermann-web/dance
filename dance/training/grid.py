# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import argparse
from pathlib import Path

from ..cli.helpers import build_config, default_cache_dir

_TUSZ_SLUGS = ("shah2018",)

_DEFAULT_GRID_NAME = {
    "korczowski2014a": "BI2014A",
    "tangermann2012": "BNCI2014",
    "shah2018": "TUSZ",
}


def run_grid_from_args(args: argparse.Namespace) -> int:
    """Dispatch `--submit` / `--local` to SLURM / in-process."""
    is_tusz = args.dataset in _TUSZ_SLUGS

    # Pick sensible per-dataset defaults if the user didn't override.
    seeds = args.seeds or ([33, 34, 35] if is_tusz else [33])
    folds = args.folds

    grid_name = args.grid_name or _DEFAULT_GRID_NAME.get(
        args.dataset, args.dataset.upper()
    )
    savedir = (
        Path(args.folder)
        if args.folder
        else (
            default_cache_dir() / "results"
            if not args.cache_folder
            else Path(args.cache_folder) / "results"
        )
    )
    savedir.mkdir(parents=True, exist_ok=True)

    base = build_config(
        args.dataset,
        debug=False,
        epochs=args.epochs,
        gpus=args.gpus or (8 if is_tusz else 1),
        seed=None,
        folder=str(savedir),
        cache_folder=args.cache_folder,
        study_path=args.study_path,
    )
    # Grid runs always use full data — never a smoke-test query that may
    # live on a dataset YAML for local iteration.
    base["data.study.query"] = None

    wandb_config = {"project": args.project, "group": grid_name, "log_model": False}
    if args.wandb_host:
        wandb_config["host"] = args.wandb_host
    base["wandb_config"] = wandb_config

    if is_tusz:
        base["infra.tasks_per_node"] = 8
        base["strategy"] = "ddp_find_unused_parameters_true"
        if args.submit:
            base["infra.slurm_use_srun"] = True
        grid = {"seed": seeds}
    else:
        grid = {"seed": seeds, "data.splitter_kwargs.fold_index": folds}

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"[run] dataset={args.dataset}  grid_name={grid_name}")
    print(f"      folder={savedir / grid_name}")
    print(f"      {n_combos} configurations ({grid})")

    if args.submit:
        return _submit_to_slurm(base, grid, grid_name, args)
    return _run_locally(base, grid, grid_name)


def _submit_to_slurm(base, grid, grid_name, args) -> int:
    from neuraltrain.utils import run_grid

    from .main import Experiment

    base["infra.cluster"] = "slurm"
    base["infra.timeout_min"] = args.timeout_min
    if args.slurm_partition:
        base["infra.slurm_partition"] = args.slurm_partition

    run_grid(
        exp_cls=Experiment,
        exp_name=grid_name,
        base_config=base,
        grid=grid,
        combinatorial=True,
        infra_mode=args.mode,
        job_name_keys=["wandb_config.name"],
    )
    print(f"[run] submitted to SLURM under group {grid_name!r}.")
    return 0


def _run_locally(base, grid, grid_name) -> int:
    """Iterate the grid sequentially in-process. No SLURM required."""
    from itertools import product

    from exca import ConfDict

    from .main import Experiment

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(
        f"[run] running {len(combos)} configurations sequentially "
        f"in-process. This may take a while."
    )
    for i, combo in enumerate(combos, 1):
        overrides = dict(zip(keys, combo))
        cfg = ConfDict(base)
        cfg.update(overrides)
        print(f"\n[run] {i}/{len(combos)}: {overrides}")
        exp = Experiment(**cfg)
        exp.infra.clear_job()
        exp.run()
    print(f"\n[run] done. {len(combos)} configurations completed.")
    return 0
