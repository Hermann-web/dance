from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from dance.ecg.adapters import ecg_batch_to_dance_batch
from dance.ecg.benchmark import (
    build_cpsc2021_group_splits,
    infer_cpsc2021_subject_id,
    run_cpsc2021_logreg_cv,
)
from dance.ecg.classifier import (
    build_cpsc2021_classification_table,
    evaluate_binary_classifier,
    extract_ecg_rr_features,
    run_cpsc2021_logreg_baseline,
)
from dance.ecg.cpsc2021_score import (
    load_prediction_json,
    score_predictions_map,
    score_record,
)
from dance.ecg.data import LudbDataset, ludb_collate
from dance.ecg.datasets.cpsc2021 import episodes_from_wfdb_ann
from dance.ecg.datasets.ludb import _events_from_wfdb_ann, read_ludb_record
from dance.ecg.evaluation import (
    evaluate_rhythm_events,
    evaluate_wave_events,
    mean_matched_iou,
)
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
from dance.ecg.rhythm_data import (
    Cpsc2021Dataset,
    build_rhythm_weighted_sampler,
    cpsc2021_collate,
)
from dance.ecg.training import (
    build_cpsc2021_loader,
    build_ludb_loader,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from dance.tests.test_dance import _TinyDance


class _FakeLudbHeader:
    sig_name = ["i", "ii", "v2"]


class _FakeLudbRecord:
    fs = 500.0
    sig_name = ["i"]
    p_signal = np.arange(20, dtype=np.float32).reshape(20, 1)


class _FakeLudbAnn:
    sample = [1, 3, 5, 6, 8, 10, 11, 13, 16, 17, 99]
    symbol = ["(", "p", ")", "(", "N", ")", "(", "t", ")", "x", ")"]


class _BadFsRecord(_FakeLudbRecord):
    fs = 0.0


def _fake_ludb_wfdb(record=None, ann=None, header=None):
    return types.SimpleNamespace(
        rdheader=lambda *a, **k: header or _FakeLudbHeader(),
        rdrecord=lambda *a, **k: record or _FakeLudbRecord(),
        rdann=lambda *a, **k: ann or _FakeLudbAnn(),
    )


def test_ludb_reader_with_wfdb_annotations(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    out = read_ludb_record("rec_01")
    assert out["fs"] == 500.0
    assert out["signal"].shape == (20,)
    assert [e.label for e in out["events"]] == ["p_wave", "qrs_complex", "t_wave"]


def test_ludb_reader_rejects_non_positive_fs(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "wfdb",
        _fake_ludb_wfdb(record=_BadFsRecord()),
    )
    with pytest.raises(ValueError, match="Invalid sampling frequency"):
        read_ludb_record("rec_01")


def test_event_conversion_skips_unknown_symbols():
    events = _events_from_wfdb_ann([1, 2, 3, 7, 8, 9], ["x", "(", ")", "(", "p", ")"])
    assert len(events) == 1
    assert events[0].label == "p_wave"


def test_ludb_reader_uses_lead_specific_annotation_extension(monkeypatch):
    seen = {}

    def _rdann(path, extension):
        seen["extension"] = extension
        return _FakeLudbAnn()

    fake = types.SimpleNamespace(
        rdheader=lambda *a, **k: _FakeLudbHeader(),
        rdrecord=lambda *a, **k: _FakeLudbRecord(),
        rdann=_rdann,
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    out = read_ludb_record("rec_01", lead="V2")
    assert seen["extension"] == "v2"
    assert out["signal"].shape == (20,)


def test_dataset_collate_and_adapter(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    ds = LudbDataset(root=tmp_path, record_ids=["rec_01", "rec_02"])
    batch = ludb_collate([ds[0], ds[1]])
    dance_batch = ecg_batch_to_dance_batch(batch)
    assert dance_batch["eeg"].shape == (2, 1, 20)
    assert dance_batch["start"].shape == (2, 3)
    assert torch.all(dance_batch["end"] >= dance_batch["start"])
    assert "channel_positions" not in dance_batch


def test_ludb_dataset_windows_and_clips_events(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    ds = LudbDataset(
        root=tmp_path,
        record_ids=["rec_01"],
        window_duration_s=0.02,
        window_stride_s=0.02,
    )
    assert len(ds) == 2
    first = ds[0]
    second = ds[1]
    assert first["eeg"].shape[-1] == 10
    assert second["eeg"].shape[-1] == 10
    assert first["event_class"].tolist() == [1, 2]
    assert second["event_class"].tolist() == [3]


def test_ecg_metric_scores_perfect_match():
    metric = EcgWaveDelineationF1(iou_threshold=0.5)
    target = {"start": torch.tensor([[0.1]]), "end": torch.tensor([[0.3]]), "class": torch.tensor([[1]])}
    pred = {"start": torch.tensor([[0.1]]), "end": torch.tensor([[0.3]]), "class": torch.tensor([[1]])}
    metric.update(as_event_lists(pred, scores=torch.tensor([[0.9]])), as_event_lists(target))
    assert metric.compute().item() == 1.0


def test_ecg_forward_pass_smoke(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    ds = LudbDataset(root=tmp_path, record_ids=["rec_01", "rec_02"])
    batch = ecg_batch_to_dance_batch(ludb_collate([ds[0], ds[1]]))
    model = _TinyDance(
        n_channels=1,
        n_classes=4,
        n_queries=8,
        duration=1.0,
        use_channel_merger=False,
    )
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


def test_wave_evaluation_reports_samplewise_metrics():
    pred = [[(0.1, 0.3, 1, 0.9), (0.4, 0.6, 2, 0.8), (0.7, 0.9, 3, 0.7)]]
    tgt = [[(0.1, 0.3, 1), (0.4, 0.6, 2), (0.7, 0.9, 3)]]
    metrics = evaluate_wave_events(
        pred,
        tgt,
        duration=1.0,
        n_samples_per_window=[100],
    )
    assert metrics["event_f1"] == pytest.approx(1.0)
    assert metrics["sample_accuracy"] == pytest.approx(1.0)
    assert metrics["sample_macro_f1"] == pytest.approx(1.0)


def test_ecg_training_entrypoint_one_epoch(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    loader = build_ludb_loader(tmp_path, ["rec_01", "rec_02"], batch_size=2, shuffle=False)
    model = _TinyDance(
        n_channels=1,
        n_classes=4,
        n_queries=8,
        duration=1.0,
        use_channel_merger=False,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optim)
    assert loss >= 0.0


def test_save_load_checkpoint_and_evaluate_model(monkeypatch, tmp_path):
    from dance.dance import Dance

    monkeypatch.setitem(__import__("sys").modules, "wfdb", _fake_ludb_wfdb())
    loader = build_ludb_loader(tmp_path, ["rec_01"], batch_size=1, shuffle=False)
    model = Dance(
        n_channels=1,
        n_classes=4,
        n_queries=8,
        duration=1.0,
        use_channel_merger=False,
    )
    checkpoint_path = save_checkpoint(model, tmp_path / "ludb.ckpt", task="ludb")
    reloaded, metadata = load_checkpoint(checkpoint_path)
    assert metadata["task"] == "ludb"
    assert reloaded.n_queries == model.n_queries

    class _OracleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, batch):
            bsz, max_events = batch["class"].shape
            n_classes = 4
            logits = torch.full((bsz, max_events, n_classes), -10.0)
            logits.scatter_(2, batch["class"].unsqueeze(-1), 10.0)
            return {
                "pred_class": logits,
                "pred_start": batch["start"],
                "pred_end": batch["end"],
            }

    metrics = evaluate_model(_OracleModel(), loader, duration=1.0, task="ludb")
    assert metrics["event_f1"] == pytest.approx(1.0)


def test_cli_ecg_ludb_train_command(monkeypatch):
    from dance.cli.main import main

    seen = {}

    def _fake_loader(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("dance.ecg.training.build_ludb_loader", _fake_loader)

    class _FakeModel:
        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(()))]

        def state_dict(self):
            return {"w": torch.zeros(())}

    monkeypatch.setattr("dance.dance.Dance", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(
        "dance.ecg.training.train_one_epoch",
        lambda model, loader, optimizer, device="cpu": 0.123,
    )
    monkeypatch.setattr(
        "dance.ecg.training.save_checkpoint",
        lambda model, path, task: path,
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
            "--lead",
            "V2",
            "--stride",
            "2.0",
            "--checkpoint-out",
            "/tmp/ludb.ckpt",
        ]
    )
    assert rc == 0
    assert seen["lead"] == "V2"
    assert seen["duration"] == 4.0
    assert seen["stride"] == 2.0


def test_cli_ecg_ludb_eval_command(monkeypatch, capsys):
    from dance.cli.main import main

    seen = {}

    def _fake_loader(**kwargs):
        seen["loader"] = kwargs
        return object()

    monkeypatch.setattr("dance.ecg.training.build_ludb_loader", _fake_loader)
    monkeypatch.setattr(
        "dance.ecg.training.load_checkpoint",
        lambda path, map_location="cpu": (object(), {"task": "ludb"}),
    )
    monkeypatch.setattr(
        "dance.ecg.training.evaluate_model",
        lambda model, loader, duration, task, device="cpu": {
            "event_f1": 0.75,
            "sample_accuracy": 0.9,
        },
    )
    rc = main(
        [
            "ecg-ludb-eval",
            "--root",
            "/tmp/ludb",
            "--records",
            "rec_01",
            "--checkpoint",
            "/tmp/ludb.ckpt",
            "--lead",
            "V2",
        ]
    )
    assert rc == 0
    assert seen["loader"]["lead"] == "V2"
    assert "\"event_f1\": 0.75" in capsys.readouterr().out


def test_cpsc_episode_merge_and_metrics():
    episodes = episodes_from_wfdb_ann(
        samples=[100, 200, 210, 320],
        aux_notes=["(AFIB", "(N", "(AFL", "(N"],
        fs=100.0,
        signal_length=500,
        global_rhythm="paroxysmal atrial fibrillation",
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


def test_cpsc_episode_parsing_handles_unbalanced_markers():
    episodes = episodes_from_wfdb_ann(
        samples=[5, 10, 20, 30, 45],
        aux_notes=["(N", "(AFIB", "(AFIB", "(N", "(N"],
        fs=100.0,
        signal_length=50,
        global_rhythm="paroxysmal atrial fibrillation",
    )
    # leading close ignored; nested open resets start; trailing unmatched close ignored
    assert len(episodes) == 1
    assert episodes[0].onset == 20
    assert episodes[0].offset == 30


def test_rhythm_evaluation_reports_matched_iou():
    pred = [[(1.0, 5.0, 1, 0.9), (8.0, 10.0, 1, 0.7)]]
    tgt = [[(1.0, 5.0, 1), (8.5, 10.0, 1)]]
    metrics = evaluate_rhythm_events(pred, tgt)
    assert metrics["episode_f1"] == pytest.approx(1.0)
    assert metrics["mean_matched_iou"] == pytest.approx((1.0 + 0.75) / 2.0)
    assert mean_matched_iou(pred, tgt) == pytest.approx((1.0 + 0.75) / 2.0)


def test_cpsc_persistent_af_defaults_to_full_record():
    episodes = episodes_from_wfdb_ann(
        samples=[],
        aux_notes=[],
        fs=200.0,
        signal_length=1000,
        global_rhythm="persistent atrial fibrillation",
    )
    assert len(episodes) == 1
    assert episodes[0].onset == 0
    assert episodes[0].offset == 999


def test_cpsc_dataset_collate_and_weighted_sampler(monkeypatch, tmp_path):
    class _CRecord:
        fs = 100.0
        sig_name = ["I"]
        p_signal = np.arange(40, dtype=np.float32).reshape(40, 1)
        comments = ["paroxysmal atrial fibrillation"]

    class _CAnnPos:
        sample = [10, 30]
        aux_note = ["(AFIB", "(N"]

    class _CAnnNeg:
        sample = []
        aux_note = []

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
        comments = ["paroxysmal atrial fibrillation"]

    class _CAnn:
        sample = [10, 30]
        aux_note = ["(AFIB", "(N"]

    fake = types.SimpleNamespace(
        rdrecord=lambda *a, **k: _CRecord(),
        rdann=lambda *a, **k: _CAnn(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)

    loader = build_cpsc2021_loader(tmp_path, ["rec_01"], batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["eeg"].shape == (1, 1, 40)

    from dance.cli.main import main

    seen = {}

    def _fake_loader(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("dance.ecg.training.build_cpsc2021_loader", _fake_loader)

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
            "--lead",
            "1",
            "--stride",
            "12.0",
        ]
    )
    assert rc == 0
    assert seen["use_weighted_sampler"] is True
    assert seen["duration"] == 30.0
    assert seen["stride"] == 12.0

    monkeypatch.setattr(
        "dance.ecg.training.load_checkpoint",
        lambda path, map_location="cpu": (object(), {"task": "cpsc2021"}),
    )
    monkeypatch.setattr(
        "dance.ecg.training.evaluate_model",
        lambda model, loader, duration, task, device="cpu": {
            "episode_f1": 0.5,
            "mean_matched_iou": 0.6,
        },
    )
    rc = main(
        [
            "ecg-cpsc2021-eval",
            "--root",
            "/tmp/cpsc2021",
            "--records",
            "A001",
            "--checkpoint",
            "/tmp/cpsc2021.ckpt",
        ]
    )
    assert rc == 0


def test_extract_ecg_rr_features_returns_finite_values():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs, dtype=np.float64)
    signal = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    signal[::100] += 1.0
    features = extract_ecg_rr_features(signal, fs)
    assert features.shape[0] >= 10
    assert np.isfinite(features).all()


def test_cpsc_classification_table_labels_windows(monkeypatch, tmp_path):
    class _CRecord:
        fs = 10.0
        sig_name = ["I"]
        p_signal = np.arange(60, dtype=np.float32).reshape(60, 1)
        comments = ["paroxysmal atrial fibrillation"]

    class _CAnn:
        sample = [10, 30]
        aux_note = ["(AFIB", "(N"]

    fake = types.SimpleNamespace(
        rdrecord=lambda *a, **k: _CRecord(),
        rdann=lambda *a, **k: _CAnn(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    table = build_cpsc2021_classification_table(
        tmp_path,
        ["rec_01"],
        window_duration_s=2.0,
        window_stride_s=2.0,
    )
    assert table["X"].shape[0] == 3
    assert table["y"].tolist() == [1, 1, 0]


def test_binary_classifier_metrics_are_reported():
    metrics = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_cpsc_logreg_baseline_smoke(monkeypatch, tmp_path):
    fs = 10.0

    def _fake_read(record_path, lead=0):
        rid = str(record_path).split("/")[-1]
        length = 120
        signal = np.zeros(length, dtype=np.float32)
        if "af" in rid:
            signal[::10] = 2.0
            episodes = [types.SimpleNamespace(label="af_episode", onset=0, offset=60)]
        else:
            signal[::20] = 1.0
            episodes = []
        return {
            "record_id": rid,
            "signal": signal,
            "fs": fs,
            "episodes": episodes,
        }

    monkeypatch.setattr("dance.ecg.classifier.read_cpsc2021_record", _fake_read)
    result = run_cpsc2021_logreg_baseline(
        root=tmp_path,
        train_record_ids=["af_train", "non_train"],
        test_record_ids=["af_test", "non_test"],
        window_duration_s=3.0,
        window_stride_s=3.0,
        max_iter=200,
    )
    assert result["n_train_windows"] == 8
    assert result["n_test_windows"] == 8
    assert result["accuracy"] >= 0.5


def test_cli_cpsc_logreg_command(monkeypatch, capsys):
    from dance.cli.main import main

    seen = {}

    def _fake_run(**kwargs):
        seen.update(kwargs)
        return {"accuracy": 0.75, "feature_names": ["x"]}

    monkeypatch.setattr("dance.ecg.classifier.run_cpsc2021_logreg_baseline", _fake_run)
    rc = main(
        [
            "ecg-cpsc2021-logreg",
            "--root",
            "/tmp/cpsc2021",
            "--train-records",
            "A001",
            "A002",
            "--test-records",
            "A003",
            "--lead",
            "V1",
            "--duration",
            "30.0",
            "--stride",
            "15.0",
        ]
    )
    assert rc == 0
    assert seen["lead"] == "V1"
    assert seen["window_duration_s"] == 30.0
    assert seen["window_stride_s"] == 15.0
    assert "\"accuracy\": 0.75" in capsys.readouterr().out


def test_cpsc_reader_rejects_non_positive_fs(monkeypatch):
    class _CAnn:
        sample = [10, 30]
        aux_note = ["(AFIB", "(N"]

    class _CBadRecord(_BadFsRecord):
        comments = ["paroxysmal atrial fibrillation"]

    fake = types.SimpleNamespace(rdrecord=lambda *a, **k: _CBadRecord(), rdann=lambda *a, **k: _CAnn())
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    from dance.ecg.datasets.cpsc2021 import read_cpsc2021_record

    with pytest.raises(ValueError, match="Invalid sampling frequency"):
        read_cpsc2021_record("A001")


def test_shared_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        ludb_collate([])
    with pytest.raises(ValueError):
        cpsc2021_collate([])


def test_adapter_validation_rejects_invalid_event_order():
    batch = {
        "eeg": torch.zeros(1, 1, 10),
        "start": torch.tensor([[0.8]]),
        "end": torch.tensor([[0.2]]),
        "class": torch.tensor([[1]]),
    }
    with pytest.raises(ValueError, match="end < start"):
        ecg_batch_to_dance_batch(batch)


def test_adapter_can_synthesize_channel_positions_when_requested():
    batch = {
        "eeg": torch.zeros(1, 2, 10),
        "start": torch.tensor([[0.1]]),
        "end": torch.tensor([[0.2]]),
        "class": torch.tensor([[1]], dtype=torch.long),
    }
    out = ecg_batch_to_dance_batch(batch, synthesize_channel_positions=True)
    assert out["channel_positions"].shape == (1, 2, 2)


def test_adapter_validation_rejects_out_of_range_boundaries():
    batch = {
        "eeg": torch.zeros(1, 1, 10),
        "start": torch.tensor([[-0.1]]),
        "end": torch.tensor([[0.2]]),
        "class": torch.tensor([[1]]),
    }
    with pytest.raises(ValueError, match="normalized to \\[0, 1\\]"):
        ecg_batch_to_dance_batch(batch)


def test_adapter_validation_rejects_non_long_class_dtype():
    batch = {
        "eeg": torch.zeros(1, 1, 10),
        "start": torch.tensor([[0.1]]),
        "end": torch.tensor([[0.2]]),
        "class": torch.tensor([[1.0]], dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="must be torch.long"):
        ecg_batch_to_dance_batch(batch)


def test_adapter_validation_rejects_bad_channel_positions_shape():
    batch = {
        "eeg": torch.zeros(2, 1, 10),
        "start": torch.tensor([[0.1], [0.2]]),
        "end": torch.tensor([[0.2], [0.3]]),
        "class": torch.tensor([[1], [1]], dtype=torch.long),
        "channel_positions": torch.zeros(2, 3, 2),
    }
    with pytest.raises(ValueError, match="channel_positions must match eeg"):
        ecg_batch_to_dance_batch(batch)


def test_adapter_validation_rejects_bad_channel_positions_rank():
    batch = {
        "eeg": torch.zeros(2, 1, 10),
        "start": torch.tensor([[0.1], [0.2]]),
        "end": torch.tensor([[0.2], [0.3]]),
        "class": torch.tensor([[1], [1]], dtype=torch.long),
        "channel_positions": torch.zeros(2, 1),
    }
    with pytest.raises(ValueError, match="shape \\(B, C, 2\\)"):
        ecg_batch_to_dance_batch(batch)


def test_cli_parse_lead_helper():
    from dance.cli.main import _parse_lead

    assert _parse_lead("1") == 1
    assert _parse_lead("-2") == -2
    assert _parse_lead("V1") == "V1"
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse_lead("   ")
    with pytest.raises(ValueError, match="not valid"):
        _parse_lead("1.0")


def test_cli_train_rejects_non_positive_numeric_args():
    from dance.cli.main import _run_ecg_train

    with pytest.raises(ValueError):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=["a"],
            lead=0,
            batch_size=0,
            lr=1e-3,
            epochs=1,
            duration=1.0,
            stride=1.0,
            n_queries=8,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )


def test_train_one_epoch_rejects_empty_loader():

    class _EmptyLoader:
        def __len__(self):
            return 0

        def __iter__(self):
            return iter(())

    model = _TinyDance(n_channels=1, n_classes=4, n_queries=8, duration=1.0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="empty loader"):
        train_one_epoch(model, _EmptyLoader(), optim)


def test_loader_builders_reject_empty_record_ids(tmp_path):
    from dance.cli.main import _run_ecg_train

    with pytest.raises(ValueError, match="at least one record id"):
        build_ludb_loader(tmp_path, [], batch_size=1)
    with pytest.raises(ValueError, match="at least one record id"):
        build_cpsc2021_loader(tmp_path, [], batch_size=1)
    with pytest.raises(ValueError):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=["a"],
            lead=0,
            batch_size=1,
            lr=1e-3,
            epochs=0,
            duration=1.0,
            stride=1.0,
            n_queries=8,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )
    with pytest.raises(ValueError):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=["a"],
            lead=0,
            batch_size=1,
            lr=1e-3,
            epochs=1,
            duration=1.0,
            stride=1.0,
            n_queries=0,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )
    with pytest.raises(ValueError):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=["a"],
            lead=0,
            batch_size=1,
            lr=0.0,
            epochs=1,
            duration=1.0,
            stride=1.0,
            n_queries=8,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )
    with pytest.raises(ValueError):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=["a"],
            lead=0,
            batch_size=1,
            lr=1e-3,
            epochs=1,
            duration=0.0,
            stride=1.0,
            n_queries=8,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )
    with pytest.raises(ValueError, match="records must contain at least one"):
        _run_ecg_train(
            task="ludb",
            root="/tmp/x",
            records=[],
            lead=0,
            batch_size=1,
            lr=1e-3,
            epochs=1,
            duration=1.0,
            stride=1.0,
            n_queries=8,
            device="cpu",
            n_classes=2,
            build_loader=lambda **kwargs: object(),
        )


def test_infer_cpsc2021_subject_id():
    assert infer_cpsc2021_subject_id("Training_set_I/data_31_1") == "data_31"
    assert infer_cpsc2021_subject_id("data_31_2") == "data_31"
    assert infer_cpsc2021_subject_id("A001") == "A001"


def test_cpsc_group_splits_keep_subjects_disjoint(monkeypatch, tmp_path):
    def _fake_read(record_path, lead=0):
        rid = Path(record_path).name
        positive = rid in {"data_1_1", "data_3_1"}
        return {
            "record_id": rid,
            "signal": np.zeros(10, dtype=np.float32),
            "fs": 10.0,
            "episodes": [types.SimpleNamespace(label="af_episode", onset=0, offset=9)] if positive else [],
        }

    monkeypatch.setattr("dance.ecg.benchmark.read_cpsc2021_record", _fake_read)
    folds = build_cpsc2021_group_splits(
        tmp_path,
        ["data_1_1", "data_1_2", "data_2_1", "data_2_2", "data_3_1", "data_3_2"],
        n_splits=3,
        random_state=7,
    )
    assert len(folds) == 3
    for fold in folds:
        assert set(fold["train_subjects"]).isdisjoint(set(fold["test_subjects"]))


def test_cpsc_logreg_cv_aggregates_fold_results(monkeypatch, tmp_path):
    def _fake_read(record_path, lead=0):
        rid = Path(record_path).name
        positive = rid in {"data_1_1", "data_3_1"}
        return {
            "record_id": rid,
            "signal": np.zeros(10, dtype=np.float32),
            "fs": 10.0,
            "episodes": [types.SimpleNamespace(label="af_episode", onset=0, offset=9)] if positive else [],
        }

    def _fake_run(**kwargs):
        score = 1.0 if any("data_1" in record for record in kwargs["test_record_ids"]) else 0.5
        return {
            "accuracy": score,
            "sensitivity": score,
            "specificity": score,
            "precision": score,
            "f1": score,
            "auroc": score,
            "average_precision": score,
            "n_train_windows": 10,
            "n_test_windows": 4,
            "n_train_positive_windows": 5,
            "n_test_positive_windows": 2,
            "feature_names": ["x"],
        }

    monkeypatch.setattr("dance.ecg.benchmark.read_cpsc2021_record", _fake_read)
    monkeypatch.setattr("dance.ecg.benchmark.run_cpsc2021_logreg_baseline", _fake_run)
    result = run_cpsc2021_logreg_cv(
        root=tmp_path,
        record_ids=["data_1_1", "data_1_2", "data_2_1", "data_2_2", "data_3_1", "data_3_2"],
        n_splits=3,
        random_state=3,
    )
    assert result["n_splits"] == 3
    assert len(result["folds"]) == 3
    assert result["accuracy_mean"] >= 0.5


def test_cpsc_score_record_matches_official_style_logic(monkeypatch):
    class _ScoreRecord:
        fs = 200.0
        sig_len = 100
        comments = ["persistent atrial fibrillation"]

    class _ScoreAnn:
        sample = [10, 20, 30, 40, 50, 60]
        aux_note = ["(AFIB", "", "", "", "", "(N"]

    fake = types.SimpleNamespace(
        rdrecord=lambda *a, **k: _ScoreRecord(),
        rdann=lambda *a, **k: _ScoreAnn(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    result = score_record("data_31_1", [[0, 99]])
    assert result["class_true"] == 1
    assert result["class_pred"] == 1
    assert result["record_score"] == pytest.approx(3.0)


def test_score_predictions_map_and_cli(monkeypatch, tmp_path, capsys):
    class _ScoreRecord:
        fs = 200.0
        sig_len = 100
        comments = ["non atrial fibrillation"]

    class _ScoreAnn:
        sample = [10, 20, 30]
        aux_note = ["(N", "(N", "(N"]

    fake = types.SimpleNamespace(
        rdrecord=lambda *a, **k: _ScoreRecord(),
        rdann=lambda *a, **k: _ScoreAnn(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wfdb", fake)
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps({"data_10_1": []}), encoding="utf-8")
    loaded = load_prediction_json(predictions_path)
    result = score_predictions_map(tmp_path, loaded)
    assert result["n_records"] == 1
    assert result["mean_score"] == pytest.approx(1.0)

    from dance.cli.main import main

    rc = main(
        [
            "ecg-cpsc2021-score",
            "--root",
            str(tmp_path),
            "--predictions-json",
            str(predictions_path),
        ]
    )
    assert rc == 0
    assert "\"mean_score\": 1.0" in capsys.readouterr().out


def test_cli_cpsc_logreg_cv_command(monkeypatch, capsys):
    from dance.cli.main import main

    seen = {}

    def _fake_run(**kwargs):
        seen.update(kwargs)
        return {"accuracy_mean": 0.8, "folds": []}

    monkeypatch.setattr("dance.ecg.benchmark.run_cpsc2021_logreg_cv", _fake_run)
    rc = main(
        [
            "ecg-cpsc2021-logreg-cv",
            "--root",
            "/tmp/cpsc2021",
            "--records",
            "data_1_1",
            "data_2_1",
            "data_3_1",
            "data_4_1",
            "data_5_1",
            "--lead",
            "V1",
            "--n-splits",
            "5",
        ]
    )
    assert rc == 0
    assert seen["lead"] == "V1"
    assert seen["n_splits"] == 5
    assert "\"accuracy_mean\": 0.8" in capsys.readouterr().out
