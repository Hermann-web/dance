# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """One-to-one matching between decoder queries and ground-truth events.

    For every window, build a (num_queries, num_targets) cost matrix
    combining a classification term (-log p(target class), scaled by
    `weight_class`) and a localisation term (1 - IoU of intervals),
    then solve it with the Hungarian algorithm. Matched queries
    inherit the target's (start, end, class); unmatched queries are
    assigned the "no event" class (0) with zero spans.
    """

    def __init__(
        self,
        weight_class: float,
        weight_iou: float,
        window_length: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_class = weight_class
        self.weight_iou = weight_iou
        self.window_length = window_length

    @staticmethod
    def _pairwise_iou(starts1, ends1, starts2, ends2):
        inter_start = torch.max(starts1[:, None], starts2[None, :])
        inter_end = torch.min(ends1[:, None], ends2[None, :])
        inter = (inter_end - inter_start).clamp(min=0)
        union = (ends1 - starts1)[:, None] + (ends2 - starts2)[None, :] - inter
        return inter / (union + 1e-7)

    def _location_cost(self, outputs, targets, batch_idx, num_targets):
        out_start = outputs["start"][batch_idx]
        out_end = outputs["end"][batch_idx]
        tgt_start = targets["start"][batch_idx][:num_targets]
        tgt_end = targets["end"][batch_idx][:num_targets]
        return 1 - self._pairwise_iou(out_start, out_end, tgt_start, tgt_end)

    def _class_cost(self, outputs, targets, batch_idx, num_targets):
        out_logits = outputs["class"][batch_idx]
        tgt_classes = targets["class"][batch_idx][:num_targets].long()
        log_probs = out_logits.log_softmax(dim=-1)
        return self.weight_class * (-log_probs[:, tgt_classes])

    def forward(self, outputs: dict, targets: dict):
        bs, num_queries = outputs["class"].shape[:2]
        device = outputs["class"].device

        matched_preds = {
            "start": torch.zeros(bs, num_queries, device=device),
            "end": torch.zeros(bs, num_queries, device=device),
            "class": outputs["class"],
        }
        matched_targets = {
            "start": torch.zeros(bs, num_queries, device=device),
            "end": torch.zeros(bs, num_queries, device=device),
            "class": torch.zeros(bs, num_queries, dtype=torch.long, device=device),
        }

        matches: list[dict] = []
        for i in range(bs):
            tgt_classes = targets["class"][i]
            num_targets = int((tgt_classes != 0).sum().item())

            if num_targets == 0:
                matches.append(
                    {
                        "q_idx": torch.empty(0, dtype=torch.long, device=device),
                        "t_idx": torch.empty(0, dtype=torch.long, device=device),
                    }
                )
                continue

            loc_cost = self._location_cost(outputs, targets, i, num_targets)
            cls_cost = self._class_cost(outputs, targets, i, num_targets)
            total_cost = loc_cost + cls_cost

            cost_np = np.nan_to_num(
                total_cost.detach().cpu().numpy(),
                nan=1e6,
                posinf=1e6,
                neginf=1e6,
            )
            q_idx, t_idx = linear_sum_assignment(cost_np)
            q_idx = torch.as_tensor(q_idx, dtype=torch.long, device=device)
            t_idx = torch.as_tensor(t_idx, dtype=torch.long, device=device)

            matches.append({"q_idx": q_idx, "t_idx": t_idx})

            matched_preds["start"][i, q_idx] = outputs["start"][i][q_idx]
            matched_preds["end"][i, q_idx] = outputs["end"][i][q_idx]
            matched_targets["start"][i, q_idx] = targets["start"][i][t_idx]
            matched_targets["end"][i, q_idx] = targets["end"][i][t_idx]
            matched_targets["class"][i, q_idx] = targets["class"][i][t_idx].long()

        return matched_preds, matched_targets, matches
