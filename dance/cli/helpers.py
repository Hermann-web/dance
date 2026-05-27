# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

import yaml
from exca import ConfDict


def configs_dir() -> Path:
    """Resolve `dance.configs` to a real Path (never a MultiplexedPath)."""
    return Path(str(importlib.resources.files("dance.configs")))


def list_datasets() -> list[str]:
    """Return every dataset slug shipped with the package, sorted alphabetically."""
    return sorted(p.stem for p in (configs_dir() / "datasets").glob("*.yaml"))


def default_cache_dir() -> Path:
    """Per-user cache for the events DataFrame produced by `Study.run()`."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dance"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config(
    dataset: str,
    *,
    debug: bool,
    epochs: int | None,
    gpus: int | None,
    seed: int | None,
    folder: str | None,
    cache_folder: str | None,
    study_path: str | None,
) -> ConfDict:
    """Merge defaults + dataset YAML + CLI overrides into one `ConfDict`."""
    base = _load_yaml(configs_dir() / "defaults.yaml")
    ds_path = configs_dir() / "datasets" / f"{dataset}.yaml"
    if not ds_path.exists():
        raise SystemExit(
            f"Unknown dataset {dataset!r}. Available: {', '.join(list_datasets())}"
        )
    ds = _load_yaml(ds_path)

    cfg = ConfDict(base)
    cfg.update(ds)

    cache = Path(cache_folder) if cache_folder else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    studies = Path(study_path) if study_path else cache / "studies"
    studies.mkdir(parents=True, exist_ok=True)

    overrides: dict = {
        "data.study.path": str(studies),
        "data.study.infra_timelines.folder": str(cache),
        "data.study.infra_timelines.mode": "cached",
        "data.neuro.infra.folder": str(cache),
        "data.neuro.infra.mode": "cached",
        "data.neuro.infra.keep_in_ram": True,
    }
    if folder is not None:
        overrides["infra.folder"] = folder
    if epochs is not None:
        overrides["n_epochs"] = epochs
    if gpus is not None:
        overrides["infra.gpus_per_node"] = gpus
    if seed is not None:
        overrides["seed"] = seed
    if debug:
        overrides.setdefault("n_epochs", 1)
        overrides.setdefault("infra.gpus_per_node", 1)
        overrides.setdefault("infra.cluster", None)
        overrides["wandb_config"] = None

    cfg.update(overrides)
    return cfg
