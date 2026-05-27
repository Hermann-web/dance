# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import lightning.pytorch as pl
import torch
from neuralset.dataloader import Batch
from torch import nn
from torchmetrics import Metric

from .. import utils
from ..losses import ConsistencyLoss, DenseLoss, DetrLoss
from ..models.decoder import _Decoder
from ..models.encoder import _Encoder


class BrainModule(pl.LightningModule):
    """End-to-end DANCE training module."""

    def __init__(
        self,
        *,
        model: _Encoder,
        decoder: _Decoder,
        dense_head: nn.Linear,
        detr_loss: DetrLoss,
        dense_loss: DenseLoss,
        consistency_loss: ConsistencyLoss,
        metrics: dict[str, Metric],
        optimizer,
        n_classes: int,
        duration: float,
        frequency: float,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder
        self.dense_head = dense_head

        self.detr_loss = detr_loss
        self.dense_loss = dense_loss
        self.consistency_loss = consistency_loss

        self.metrics = nn.ModuleDict(
            {f"{split}_{k}": v for k, v in metrics.items() for split in ("val", "test")}
        )

        self.optimizer = optimizer
        self.n_classes = n_classes
        self.duration = duration
        self.frequency = frequency
        self.save_hyperparameters(
            config or {}, ignore=["model", "decoder", "dense_head", "metrics"]
        )

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        encoder_output = self.model(
            batch.data["neuro"],
            channel_positions=batch.data.get("channel_positions"),
        )["c_out"]
        decoder_output = self.decoder(encoder_output)
        return {
            "class": decoder_output["class"],
            "start": decoder_output["start"],
            "end": decoder_output["end"],
            "dense": self.dense_head(encoder_output),
        }

    def _run_step(self, batch: Batch, split: str):
        preds = self(batch)
        targets = {
            key: batch.data[f"{key}_target"]
            for key in ("class", "start", "end", "dense")
            if f"{key}_target" in batch.data
        }

        detr_total, detr_details = self.detr_loss(preds, targets)
        dense_term = self.dense_loss(preds, targets)
        consistency_term = self.consistency_loss(preds)
        total = detr_total + dense_term + consistency_term

        self.log(f"{split}_loss", total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_dense_loss", dense_term, on_epoch=True, sync_dist=True)
        self.log(
            f"{split}_consistency_loss", consistency_term, on_epoch=True, sync_dist=True
        )
        for name, val in detr_details.items():
            self.log(f"{split}_{name}", val, on_epoch=True, sync_dist=True)

        if split != "train":
            self._update_metrics(split, preds, targets)

        return total, preds, targets

    def _update_metrics(self, split: str, preds: dict, targets: dict) -> None:
        pred_events, gt_events = utils.extract_events_from_detr_batch(
            preds,
            targets,
            window_length=self.duration,
        )
        for name, metric in self.metrics.items():
            if not name.startswith(split):
                continue
            device = metric.device

            if "f1_sample" in name:
                pred_masks = utils.make_masks(
                    pred_events, device, self.duration, self.frequency, self.n_classes
                )
                gt_masks = utils.make_masks(
                    gt_events, device, self.duration, self.frequency, self.n_classes
                )
                metric.update(pred_masks, gt_masks)
            else:
                metric.update(pred_events, gt_events)

            self.log(
                name, metric, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )

    def training_step(self, batch: Batch, batch_idx: int):
        loss, _, _ = self._run_step(batch, "train")
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", round(lr, 6), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        _, preds, targets = self._run_step(batch, "val")
        return {"preds": preds, "targets": targets}

    def test_step(self, batch: Batch, batch_idx: int):
        _, preds, targets = self._run_step(batch, "test")
        return {"preds": preds, "targets": targets}

    def configure_optimizers(self):
        unfrozen = [p for p in self.parameters() if p.requires_grad]
        if (
            self.optimizer.scheduler is not None
            and self.optimizer.scheduler.__class__.__name__ == "OneCycleLR"
        ):
            return self.optimizer.build(
                unfrozen,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        return self.optimizer.build(unfrozen)
