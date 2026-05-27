# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import typing as tp

import torch
import torch.nn.functional as F
from neuraltrain.losses.base import BaseLoss
from torch import nn

from . import utils
from .matcher import HungarianMatcher


class IoULoss(BaseLoss):
    """1 - IoU between predicted and ground-truth 1-D intervals, averaged
    over the batch. Used as the localisation term of the DETR loss.

    YAML usage:

        iou_loss:
          name: IoULoss
          mode: start_end
    """

    mode: tp.Literal["start_end", "center_duration"] = "start_end"

    def build(self):
        mode = self.mode

        def iou_loss(*args, window_length: float | None = None) -> torch.Tensor:
            if mode == "center_duration":
                pc, pd, tc, td = args
                ps = (pc - 0.5 * pd) * window_length
                pe = (pc + 0.5 * pd) * window_length
                ts = (tc - 0.5 * td) * window_length
                te = (tc + 0.5 * td) * window_length
            else:
                ps, pe, ts, te = args
            inter = torch.clamp(torch.min(pe, te) - torch.max(ps, ts), min=0.0)
            union = torch.clamp(torch.max(pe, te) - torch.min(ps, ts), min=1e-6)
            return (1.0 - inter / union).mean()

        return iou_loss


class DetrLoss(nn.Module):
    """Classification + IoU loss on Hungarian-matched query / target pairs.

    The matcher pairs each ground-truth event with one decoder query.
    The classification term is then applied to all queries (matched
    ones supervised against their target class, unmatched against the
    "no event" class), and the IoU term is applied only to the matched
    pairs.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        class_loss: nn.Module,
        iou_loss: tp.Callable[..., torch.Tensor],
        *,
        weight_class: float,
        weight_iou: float,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.class_loss = class_loss
        self.iou_loss = iou_loss
        self.weight_class = weight_class
        self.weight_iou = weight_iou

    def forward(self, preds: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        matched_preds, matched_targets, _ = self.matcher(preds, targets)

        logits = matched_preds["class"].reshape(-1, preds["class"].shape[-1])
        if isinstance(self.class_loss, nn.CrossEntropyLoss):
            labels = matched_targets["class"].reshape(-1).long()
        else:
            labels = F.one_hot(
                matched_targets["class"].reshape(-1).long(),
                num_classes=preds["class"].shape[-1],
            ).float()
        cls_term = self.weight_class * self.class_loss(logits, labels)

        iou_term = self.weight_iou * self.iou_loss(
            matched_preds["start"],
            matched_preds["end"],
            matched_targets["start"],
            matched_targets["end"],
        )

        total = cls_term + iou_term
        return total, {"class_loss": cls_term.detach(), "iou_loss": iou_term.detach()}


class DenseLoss(nn.Module):
    """Cross-entropy on the dense head: a per-timestep classifier over
    the same `n_classes` as the DETR head.
    """

    def __init__(self, ce_loss: nn.Module, *, weight: float) -> None:
        super().__init__()
        self.ce_loss = ce_loss
        self.weight = weight

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        logits = preds["dense"].view(-1, preds["dense"].shape[-1])
        labels = targets["dense"].squeeze(1).long().view(-1)
        return self.weight * self.ce_loss(logits, labels)


class ConsistencyLoss(nn.Module):
    """KL divergence between the dense head's per-timestep softmax and
    a dense probability map derived from the DETR queries by rendering
    each predicted (start, end, class) span back over the time axis.
    Couples the two heads so they agree on the per-timestep distribution.
    """

    def __init__(
        self,
        kl_loss: nn.Module,
        *,
        n_classes: int,
        duration: float,
        frequency: float,
        weight: float,
    ) -> None:
        super().__init__()
        self.kl_loss = kl_loss
        self.n_classes = n_classes
        self.duration = duration
        self.frequency = frequency
        self.weight = weight

    def forward(self, preds: dict) -> torch.Tensor:
        dense_probs = torch.softmax(preds["dense"], dim=-1).clamp(min=1e-8)
        detr_probs = utils.detr_to_dense_probs(
            preds, self.duration, self.frequency, self.n_classes
        ).clamp(min=1e-8)
        return (
            self.weight * self.kl_loss(detr_probs.log(), dense_probs).sum(dim=-1).mean()
        )
