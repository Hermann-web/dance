# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import json
import re
from pathlib import Path

from lightning.pytorch.callbacks import Callback

from .. import utils


class PredictedEvents(Callback):
    """Dump per-window subjects, predicted events and ground-truth events
    to JSON for offline per-subject metric computation.

    Test predictions are saved at the end of testing. Validation
    predictions are saved only at the end of training (single snapshot
    from the last validation epoch) to keep the cost negligible.
    """

    _SUBJECT_RE = re.compile(r"subject=(\w+)")

    def __init__(self) -> None:
        self._batches: dict[str, dict[int, dict]] = {"val": {}, "test": {}}

    @staticmethod
    def _resolve_subject(segment) -> str | None:
        events = getattr(segment, "ns_events", None) or []
        if not events:
            return None
        event = events[0]
        subj = getattr(event, "subject", None)
        if subj is None:
            timeline = getattr(event, "timeline", None)
            if timeline is not None:
                m = PredictedEvents._SUBJECT_RE.search(str(timeline))
                subj = m.group(1) if m else None
        return str(subj) if subj is not None else None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._batches["val"] = {}

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._batches["test"] = {}

    def _collect(self, split: str, batch, batch_idx: int, outputs, pl_module) -> None:
        if outputs is None:
            return
        pred_events, gt_events = utils.extract_events_from_detr_batch(
            outputs["preds"],
            outputs["targets"],
            window_length=getattr(pl_module, "duration", 0.0),
        )
        self._batches[split][batch_idx] = {
            "subjects": [self._resolve_subject(s) for s in batch.segments],
            "pred_events": pred_events,
            "gt_events": gt_events,
        }

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect("val", batch, batch_idx, outputs, pl_module)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect("test", batch, batch_idx, outputs, pl_module)

    def _save(self, trainer, split: str) -> None:
        if trainer.logger is None or trainer.logger.save_dir is None:
            return
        out = Path(trainer.logger.save_dir) / "callbacks"
        out.mkdir(parents=True, exist_ok=True)
        suffix = f"_rank{trainer.global_rank}" if trainer.world_size > 1 else ""
        path = out / f"{split}_batch_predictions{suffix}.json"
        with open(path, "w") as f:
            json.dump(self._batches[split], f)

    def on_train_end(self, trainer, pl_module) -> None:
        if self._batches["val"]:
            self._save(trainer, "val")

    def on_test_end(self, trainer, pl_module) -> None:
        self._save(trainer, "test")
