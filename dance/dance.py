# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
from torch import nn

from .losses import ConsistencyLoss, DenseLoss, DetrLoss, IoULoss
from .matcher import HungarianMatcher
from .models.decoder import Decoder
from .models.encoder import Encoder


class Dance(nn.Module):
    """Standalone DANCE model.

    Bundles the encoder (CNN + Perceiver), the DETR-style decoder, the
    dense head and the full loss stack into a single nn.Module. Pass a
    batch dict, get back predictions and (when targets are present) the
    training loss in one call.

    The architecture is locked to the paper configuration; defaults
    live with each component (`dance.models.encoder.Encoder`,
    `dance.models.decoder.Decoder`). To tweak any architecture or loss
    hyperparameter, subclass `Dance` and override the relevant
    construction (pass a custom `Encoder(...)` / `Decoder(...)` or
    re-instantiate the loss modules with different `weight*` kwargs).

    Example:

        model = Dance(n_channels=16, n_classes=3, n_queries=150, duration=32.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        for batch in your_dataloader:
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

    Expected batch keys (all tensors):

    -- INPUTS --
    - eeg: (B, n_channels, T) float, raw EEG.
    - channel_positions: (B, n_channels, 2) float, normalised (x, y)
      electrode coordinates. Required only if `use_channel_merger=True`.

    When training the model, the batch must also include target
    tensors so the loss can be computed:

    -- TARGETS --
    - start, end: (B, max_events) float in [0, 1], event spans
      normalised to the window (0 = window start, 1 = window end);
      zero-pad unused slots.
    - class: (B, max_events) long, class id per event
      (0 = padding / no-event), values in [0, n_classes).
    - dense: (B, num_latents) long, per-timestep target class id.
      OPTIONAL — derived from (start, end, class) if absent. Pass it
      yourself only if you want a different rendering policy.

    Note that `max_events` should be slightly lower than the chosen
    `n_queries` to obtain good performances.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_classes: int,
        n_queries: int,
        duration: float,
        use_channel_merger: bool = True,
    ) -> None:
        """Instantiate a paper-configured DANCE model.

        Parameters
        ----------
        n_channels
            Number of EEG channels per window.
        n_classes
            Number of event classes including background. Background is
            class id 0; "real" classes are 1..n_classes-1.
        n_queries
            Number of DETR decoder slots, i.e. the maximum number of
            events the model can predict per window. Set it slightly
            above the worst-case event count per window for your task.
        duration
            Window length in seconds.
        use_channel_merger
            If True (default, paper config), the ChannelMerger projects
            the raw channels to a fixed number of virtual ones via
            spatial Fourier attention; the batch must then include
            `channel_positions`. If False, the conv stack consumes the
            raw channels directly and `channel_positions` is no longer
            needed.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.duration = duration
        self.use_channel_merger = use_channel_merger

        encoder_cfg = Encoder()
        if not use_channel_merger:
            encoder_cfg.encoder_config.merger_config = None
        dim = encoder_cfg.dim
        self.num_latents = encoder_cfg.perceiver_config.num_latents
        # One token per latent, so the per-token rate is num_latents / duration.
        self.frequency = self.num_latents / duration
        self.encoder = encoder_cfg.build(n_in_channels=n_channels, n_outputs=dim)
        self.decoder = Decoder(n_queries=n_queries, n_classes=n_classes).build(
            n_in_channels=dim
        )
        self.dense_head = nn.Linear(dim, n_classes)

        # A Hungarian matcher pairs each ground-truth event with one decoder query
        # then DetrLoss supervises the matched (class, span) pairs while DenseLoss
        # and ConsistencyLoss tie the dense head to the same per-timestep distribution.
        matcher = HungarianMatcher(
            weight_class=1.0, weight_iou=5.0, window_length=duration
        )
        self.detr_loss = DetrLoss(
            matcher=matcher,
            class_loss=nn.CrossEntropyLoss(),
            iou_loss=IoULoss(mode="start_end").build(),
            weight_class=1.0,
            weight_iou=5.0,
        )
        self.dense_loss = DenseLoss(nn.CrossEntropyLoss(), weight=1.0)
        self.consistency_loss = ConsistencyLoss(
            nn.KLDivLoss(reduction="none"),
            n_classes=n_classes,
            duration=duration,
            frequency=self.frequency,
            weight=0.5,
        )

    def _dense_from_events(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        cls: torch.Tensor,
    ) -> torch.Tensor:
        """Render per-event spans into the (B, 1, num_latents) class map
        that DenseLoss + ConsistencyLoss expect.
        `start` and `end` are in [0, 1] (window-relative); class id 0
        marks padding / no-event.
        """
        B = start.shape[0]
        T = self.num_latents
        dense = torch.zeros((B, 1, T), dtype=torch.long, device=start.device)
        # Discretise once: (B, max_events) of [0, T) token indices.
        s_tok = (start * T).clamp(0, T).long()
        e_tok = (end * T).clamp(0, T).long()
        for b in range(B):
            for i in range(start.shape[1]):
                c = int(cls[b, i])
                if c == 0:
                    continue
                s, e = int(s_tok[b, i]), int(e_tok[b, i])
                if s < e:
                    dense[b, 0, s:e] = c
        return dense

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "eeg" not in batch and "signal" in batch:
            batch = {**batch, "eeg": batch["signal"]}
        encoder_output = self.encoder(
            batch["eeg"],
            channel_positions=batch.get("channel_positions"),
        )["c_out"]
        decoder_output = self.decoder(encoder_output)
        dense_logits = self.dense_head(encoder_output)

        preds = {
            "class": decoder_output["class"],
            "start": decoder_output["start"],
            "end": decoder_output["end"],
            "dense": dense_logits,
        }
        out: dict[str, torch.Tensor] = {
            "pred_class": preds["class"],
            "pred_start": preds["start"],
            "pred_end": preds["end"],
            "pred_dense": preds["dense"],
        }

        # No targets -> inference path; return predictions only.
        event_keys = {"start", "end", "class"}
        if not event_keys.issubset(batch.keys()):
            return out

        # With the target, also return the loss for training. The dense
        # target is just the per-token rendering of (start, end, class);
        # accept it pre-computed in the batch, otherwise derive it.
        if "dense" in batch:
            dense_target = batch["dense"]
            if dense_target.ndim == 2:
                dense_target = dense_target.unsqueeze(1)
        else:
            dense_target = self._dense_from_events(
                batch["start"], batch["end"], batch["class"]
            )
        targets = {
            "start": batch["start"],
            "end": batch["end"],
            "class": batch["class"],
            "dense": dense_target,
        }

        detr_total, detr_details = self.detr_loss(preds, targets)
        dense_term = self.dense_loss(preds, targets)
        consistency_term = self.consistency_loss(preds)
        total = detr_total + dense_term + consistency_term

        out["loss"] = total
        out["loss_details"] = {
            "detr_class": detr_details["class_loss"],
            "detr_iou": detr_details["iou_loss"],
            "dense": dense_term.detach(),
            "consistency": consistency_term.detach(),
        }
        return out
