# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Top-level pytest fixtures."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tmp_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped writable cache directory for exca / pooch / mne / wandb."""
    cache = tmp_path_factory.mktemp("dance_cache")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MNE_DATA", str(cache / "mne"))
    return cache


@pytest.fixture
def isolate_cwd(tmp_path: Path) -> Path:
    """Run a test in its own scratch dir so it can write callback dumps freely."""
    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td)
        previous = Path.cwd()
        try:
            os.chdir(cwd)
            yield cwd
        finally:
            os.chdir(previous)
