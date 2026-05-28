from __future__ import annotations

import numpy as np
import torch
import types

from dance.ecg.adapters import ecg_batch_to_dance_batch
from dance.ecg.data import LudbDataset, ludb_collate
from dance.ecg.datasets.ludb import _events_from_wfdb_ann, read_ludb_record
from dance.ecg.metrics import (
    EcgOffsetMAE,
    EcgOnsetMAE,
    EcgToleranceF1,
    EcgWaveDelineationF1,
    as_event_lists,
)
from dance.ecg.training import build_ludb_loader, train_one_epoch
from dance.tests.test_dance import _TinyDance


class _FakeRecord:
    fs = 500.0
    sig_name = ["I"]
    p_signal = np.arange(20, dtype=np.float32).reshape(20, 1)


class _FakeAnn:
    sample = [1, 5, 6, 10, 11, 16, 2, 4]
    symbol = ["(p", ")p", "(N", ")N", "(t", ")t", "(x", ")x"]


def test_ludb_reader_with_wfdb_annotations(monkeypatch):
    fake = types.SimpleNamespace(rdrecord=lambda *a, **k: _FakeRecord(), rdann=lambda *a, **k: _FakeAnn())
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    out = read_ludb_record("rec_01")
    assert out["fs"] == 500.0
    assert out["signal"].shape == (20,)
    assert [e.label for e in out["events"]] == ["p_wave", "qrs_complex", "t_wave"]


def test_event_conversion_skips_unknown_symbols():
    events = _events_from_wfdb_ann([1, 5, 7, 9], ["(x", ")x", "(p", ")p"])
    assert len(events) == 1
    assert events[0].label == "p_wave"


def test_dataset_collate_and_adapter(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(rdrecord=lambda *a, **k: _FakeRecord(), rdann=lambda *a, **k: _FakeAnn())
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    ds = LudbDataset(root=tmp_path, record_ids=["rec_01", "rec_02"])
    batch = ludb_collate([ds[0], ds[1]])
    dance_batch = ecg_batch_to_dance_batch(batch)
    assert dance_batch["eeg"].shape == (2, 1, 20)
    assert dance_batch["start"].shape == (2, 3)
    assert torch.all(dance_batch["end"] >= dance_batch["start"])


def test_ecg_metric_scores_perfect_match():
    metric = EcgWaveDelineationF1(iou_threshold=0.5)
    target = {"start": torch.tensor([[0.1]]), "end": torch.tensor([[0.3]]), "class": torch.tensor([[1]])}
    pred = {"start": torch.tensor([[0.1]]), "end": torch.tensor([[0.3]]), "class": torch.tensor([[1]])}
    metric.update(as_event_lists(pred, scores=torch.tensor([[0.9]])), as_event_lists(target))
    assert metric.compute().item() == 1.0


def test_ecg_forward_pass_smoke(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(rdrecord=lambda *a, **k: _FakeRecord(), rdann=lambda *a, **k: _FakeAnn())
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    ds = LudbDataset(root=tmp_path, record_ids=["rec_01", "rec_02"])
    batch = ecg_batch_to_dance_batch(ludb_collate([ds[0], ds[1]]))
    model = _TinyDance(n_channels=1, n_classes=4, n_queries=8, duration=1.0)
    out = model(batch)
    assert out["loss"].ndim == 0


def test_ecg_onset_offset_mae():
    pred = [[(0.10, 0.31, 1, 0.9)]]
    tgt = [[(0.12, 0.29, 1)]]
    onset = EcgOnsetMAE(iou_threshold=0.1)
    offset = EcgOffsetMAE(iou_threshold=0.1)
    onset.update(pred, tgt)
    offset.update(pred, tgt)
    assert torch.isclose(onset.compute(), torch.tensor(0.02))
    assert torch.isclose(offset.compute(), torch.tensor(0.02))


def test_ecg_tolerance_f1():
    metric = EcgToleranceF1(tolerance=0.02)
    pred = [[(0.10, 0.30, 1, 0.9), (0.50, 0.70, 2, 0.8)]]
    tgt = [[(0.11, 0.31, 1), (0.48, 0.69, 2)]]
    metric.update(pred, tgt)
    assert torch.isclose(metric.compute(), torch.tensor(1.0))


def test_ecg_training_entrypoint_one_epoch(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(rdrecord=lambda *a, **k: _FakeRecord(), rdann=lambda *a, **k: _FakeAnn())
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    loader = build_ludb_loader(tmp_path, ["rec_01", "rec_02"], batch_size=2, shuffle=False)
    model = _TinyDance(n_channels=1, n_classes=4, n_queries=8, duration=1.0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optim)
    assert loss >= 0.0


def test_cli_ecg_ludb_train_command(monkeypatch):
    from dance.cli.main import main

    monkeypatch.setattr("dance.ecg.training.build_ludb_loader", lambda **kwargs: object())

    class _FakeModel:
        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(()))]

    monkeypatch.setattr("dance.dance.Dance", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(
        "dance.ecg.training.train_one_epoch",
        lambda model, loader, optimizer, device="cpu": 0.123,
    )

    rc = main(
        [
            "ecg-ludb-train",
            "--root",
            "/tmp/ludb",
            "--records",
            "rec_01",
            "rec_02",
            "--epochs",
            "1",
        ]
    )
    assert rc == 0
