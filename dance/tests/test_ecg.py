from __future__ import annotations

import numpy as np
import torch
import types

from dance.ecg.adapters import ecg_batch_to_dance_batch
from dance.ecg.data import LudbDataset, ludb_collate
from dance.ecg.datasets.ludb import _events_from_wfdb_ann, read_ludb_record
from dance.ecg.metrics import (
    EcgBurdenError,
    EcgOffsetDelay,
    EcgOffsetMAE,
    EcgOnsetDelay,
    EcgOnsetMAE,
    EcgRhythmEpisodeF1,
    EcgToleranceF1,
    EcgWaveDelineationF1,
    as_event_lists,
)
from dance.ecg.datasets.cpsc2021 import episodes_from_wfdb_ann
from dance.ecg.training import build_cpsc2021_loader, build_ludb_loader, train_one_epoch
from dance.ecg.rhythm_data import (
    Cpsc2021Dataset,
    build_rhythm_weighted_sampler,
    cpsc2021_collate,
)
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


def test_cpsc_episode_merge_and_metrics():
    episodes = episodes_from_wfdb_ann(
        samples=[100, 200, 210, 320],
        symbols=["(AFIB", ")AFIB", "(AFIB", ")AFIB"],
        fs=100.0,
        merge_gap_seconds=0.2,
    )
    assert len(episodes) == 1
    assert episodes[0].onset == 100 and episodes[0].offset == 320

    pred = [[(0.1, 0.5, 1, 0.9)]]
    tgt = [[(0.12, 0.52, 1)]]
    f1 = EcgRhythmEpisodeF1(iou_threshold=0.5)
    onset = EcgOnsetDelay()
    offset = EcgOffsetDelay()
    burden = EcgBurdenError()
    f1.update(pred, tgt)
    onset.update(pred, tgt)
    offset.update(pred, tgt)
    burden.update(pred, tgt)
    assert torch.isclose(f1.compute(), torch.tensor(1.0))
    assert torch.isclose(onset.compute(), torch.tensor(0.02))
    assert torch.isclose(offset.compute(), torch.tensor(0.02))
    assert torch.isclose(burden.compute(), torch.tensor(0.0))


def test_cpsc_dataset_collate_and_weighted_sampler(monkeypatch, tmp_path):
    class _CRecord:
        fs = 100.0
        sig_name = ["I"]
        p_signal = np.arange(40, dtype=np.float32).reshape(40, 1)

    class _CAnnPos:
        sample = [10, 30]
        symbol = ["(AFIB", ")AFIB"]

    class _CAnnNeg:
        sample = []
        symbol = []

    def _rdrecord(*args, **kwargs):
        return _CRecord()

    def _rdann(path, extension):
        return _CAnnPos() if str(path).endswith("pos") else _CAnnNeg()

    fake = types.SimpleNamespace(rdrecord=_rdrecord, rdann=_rdann)
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)

    ds = Cpsc2021Dataset(root=tmp_path, record_ids=["pos", "neg"])
    batch = cpsc2021_collate([ds[0], ds[1]])
    assert batch["eeg"].shape == (2, 1, 40)
    assert batch["class"].shape[1] >= 1
    sampler = build_rhythm_weighted_sampler(ds, positive_weight=7.0, negative_weight=1.0)
    w = sampler.weights.tolist()
    assert w[0] == 7.0 and w[1] == 1.0


def test_cpsc_loader_and_cli_command(monkeypatch, tmp_path):
    class _CRecord:
        fs = 100.0
        sig_name = ["I"]
        p_signal = np.arange(40, dtype=np.float32).reshape(40, 1)

    class _CAnn:
        sample = [10, 30]
        symbol = ["(AFIB", ")AFIB"]

    fake = types.SimpleNamespace(
        rdrecord=lambda *a, **k: _CRecord(),
        rdann=lambda *a, **k: _CAnn(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)

    loader = build_cpsc2021_loader(tmp_path, ["rec_01"], batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["eeg"].shape == (1, 1, 40)

    from dance.cli.main import main

    monkeypatch.setattr("dance.ecg.training.build_cpsc2021_loader", lambda **kwargs: object())

    class _FakeModel:
        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(()))]

    monkeypatch.setattr("dance.dance.Dance", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(
        "dance.ecg.training.train_one_epoch",
        lambda model, loader, optimizer, device="cpu": 0.321,
    )
    rc = main(
        [
            "ecg-cpsc2021-train",
            "--root",
            "/tmp/cpsc2021",
            "--records",
            "A001",
            "--epochs",
            "1",
        ]
    )
    assert rc == 0
