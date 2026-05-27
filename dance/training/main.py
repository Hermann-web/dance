# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
from exca import TaskInfra
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from neuraltrain import BaseLoss, BaseMetric, BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig
from torch import nn

from .. import losses as _losses_module  # noqa: F401  registers vendored IoULoss
from .. import metrics as _metrics_module  # noqa: F401  registers F1Event
from .. import models as _models_module  # noqa: F401  registers vendored configs
from ..losses import ConsistencyLoss, DenseLoss, DetrLoss
from ..matcher import HungarianMatcher
from ..models.decoder import Decoder
from ..utils import is_rank_zero
from . import (
    transforms as _transforms_module,  # noqa: F401  registers SeizurePreprocessor
)
from .callbacks import PredictedEvents
from .data import Data
from .pl_module import BrainModule

logger = logging.getLogger(__name__)


class Experiment(BaseExperiment):
    """End-to-end DANCE experiment configuration."""

    data: Data

    seed: int | None = 33

    brain_model_config: BaseModelConfig
    decoder_config: Decoder

    losses: dict[str, BaseLoss]
    optimizer: LightningOptimizer
    metrics: list[BaseMetric]

    wandb_config: WandbLoggerConfig | None = None

    strategy: str = "auto"
    accelerator: str = "gpu"
    precision: int = 32

    n_epochs: int = 10
    max_steps: int = -1
    val_check_interval: float | int = 1.0
    patience: int = 100

    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    accumulate_grad_batches: int = 1

    save_checkpoints: bool = False
    monitor: str = "val_f1_event"
    monitor_mode: tp.Literal["max", "min"] = "max"

    weight_iou: float = 5.0
    weight_class: float = 1.0
    weight_dense: float = 1.0
    weight_consistency: float = 0.5

    eval_only: bool = False
    train_only: bool = False
    checkpoint_path: str | None = None

    _trainer: pl.Trainer | None = None
    _module: BrainModule | None = None
    _logger: WandbLogger | None = None

    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            raise ValueError("infra.folder must be specified to save results")
        if self.infra.gpus_per_node == 1:
            self.data.num_workers = self.infra.cpus_per_task or 0
        else:
            self.data.num_workers = 16

        feat_class = self.data.features.get("feature_class")
        if feat_class is None or feat_class.mapping is None:
            raise ValueError(
                "data.features['feature_class'] with a non-empty `mapping` "
                "must be configured so the decoder knows n_classes."
            )
        self.decoder_config.n_classes = len(set(feat_class.mapping.values()))

        n_queries = self.decoder_config.n_queries
        for feat in self.data.features.values():
            if feat.max_events is not None:
                feat.max_events = n_queries

        for m in self.metrics:
            if "accuracy" in m.log_name:
                m.kwargs["num_classes"] = self.decoder_config.n_classes
            if m.log_name.endswith("f1_sample"):
                m.kwargs["num_labels"] = self.decoder_config.n_classes

    def _build_logger(self) -> WandbLogger | None:
        if self.wandb_config is None:
            return None
        return self.wandb_config.build(
            save_dir=self.infra.folder,
            xp_config=self.model_dump(),
            run_id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
        )

    def _build_trainer(self) -> pl.Trainer:
        callbacks: list = [
            EarlyStopping(
                monitor=self.monitor, mode=self.monitor_mode, patience=self.patience
            ),
            PredictedEvents(),
        ]
        if self.save_checkpoints and is_rank_zero():
            callbacks.append(
                ModelCheckpoint(
                    save_last=False,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor=self.monitor,
                    mode=self.monitor_mode,
                    save_on_train_epoch_end=False,
                )
            )

        if self.max_steps > 0:
            max_epochs = -1
            check_val_every_n_epoch = None
            val_check_interval = int(self.val_check_interval)
        else:
            max_epochs = self.n_epochs
            check_val_every_n_epoch = 1
            val_check_interval = float(self.val_check_interval)

        return pl.Trainer(
            strategy=self.strategy,
            devices=self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_steps=self.max_steps,
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            precision=self.precision,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=callbacks,
            logger=self._logger,
            default_root_dir=str(self.infra.folder),
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

    def _build_module(self, train_loader) -> BrainModule:
        batch = next(iter(train_loader))
        n_in_channels = batch.data["neuro"].shape[1]
        brain_model = self.brain_model_config.build(
            n_in_channels=n_in_channels,
            n_outputs=self.brain_model_config.dim,
        )
        decoder = self.decoder_config.build(n_in_channels=self.brain_model_config.dim)
        dense_head = nn.Linear(
            self.brain_model_config.dim,
            self.decoder_config.n_classes,
        )
        matcher = HungarianMatcher(
            weight_class=self.weight_class,
            weight_iou=self.weight_iou,
            window_length=self.data.duration,
        )

        loss_modules = {name: cfg.build() for name, cfg in self.losses.items()}
        for required in ("class_loss", "iou_loss", "dense_loss", "consistency_loss"):
            if required not in loss_modules:
                raise ValueError(f"Missing required loss {required!r} in self.losses.")

        # Token rate seen by the DETR decoder: num_latents / window_duration
        # when a Perceiver bottleneck is present, else the raw EEG rate.
        if self.brain_model_config.perceiver_config is not None:
            num_latents = self.brain_model_config.perceiver_config.num_latents
            frequency = num_latents / self.data.duration
        else:
            frequency = self.data.neuro.frequency

        detr_loss = DetrLoss(
            matcher=matcher,
            class_loss=loss_modules["class_loss"],
            iou_loss=loss_modules["iou_loss"],
            weight_class=self.weight_class,
            weight_iou=self.weight_iou,
        )
        dense_loss = DenseLoss(loss_modules["dense_loss"], weight=self.weight_dense)
        consistency_loss = ConsistencyLoss(
            loss_modules["consistency_loss"],
            n_classes=self.decoder_config.n_classes,
            duration=self.data.duration,
            frequency=frequency,
            weight=self.weight_consistency,
        )

        return BrainModule(
            model=brain_model,
            decoder=decoder,
            dense_head=dense_head,
            detr_loss=detr_loss,
            dense_loss=dense_loss,
            consistency_loss=consistency_loss,
            metrics={m.log_name: m.build() for m in self.metrics},
            optimizer=self.optimizer,
            n_classes=self.decoder_config.n_classes,
            duration=self.data.duration,
            frequency=frequency,
            config=self.model_dump(),
        )

    @infra.apply
    def run(self):
        pl.seed_everything(self.seed, workers=True)
        self._logger = self._build_logger()

        loaders = self.data.build()
        self._trainer = self._build_trainer()
        self._module = self._build_module(loaders["train"])

        if self.eval_only:
            ckpt = self.checkpoint_path
            if ckpt is None:
                raise ValueError("eval_only=True requires checkpoint_path")
            self._trainer.test(
                self._module,
                dataloaders=loaders["test"],
                ckpt_path=ckpt,
            )
            return

        self._trainer.fit(
            self._module,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )

        if self.train_only:
            return

        ckpt = (
            str(Path(self.infra.folder) / "best.ckpt") if self.save_checkpoints else None
        )
        self._trainer.test(self._module, dataloaders=loaders["test"], ckpt_path=ckpt)
