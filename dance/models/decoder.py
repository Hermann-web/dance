# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

from math import log

import torch
import torch.nn as nn
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.transformer import TransformerEncoder


def _sinusoidal_embeddings(num_positions, embedding_dim, *, device=None, dtype=None):
    pe = torch.zeros(num_positions, embedding_dim, device=device, dtype=dtype)
    position = torch.arange(num_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device)
        * (-log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if embedding_dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Decoder(BaseModelConfig):
    """DETR-style event decoder.

    A fixed number of learnable queries (`n_queries`) cross-attend the
    encoder memory and then project independently to five heads: a class
    logit over `n_classes`, and four scalar regressors (start, end,
    center, duration) constrained to [0, 1] and read as
    window-relative offsets.

    The `transformer` default (depth 4, 4 heads) matches the DANCE
    paper; override the field to swap in a different stack.
    """

    hidden_dim: int = 256
    n_queries: int = 100
    transformer: TransformerEncoder = TransformerEncoder(depth=4, heads=4)
    dropout: float = 0.1
    n_classes: int = 1

    def model_post_init(self, __context=None):
        super().model_post_init(__context)
        for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
            setattr(self.transformer, key, self.dropout)

    def build(self, n_in_channels: int) -> nn.Module:
        return _Decoder(config=self, n_in_channels=n_in_channels)


class _Decoder(nn.Module):
    def __init__(self, config: Decoder, n_in_channels: int) -> None:
        super().__init__()
        self.config = config

        self.n_in_channels = n_in_channels or config.hidden_dim
        if self.n_in_channels != config.hidden_dim:
            self.input_proj: nn.Module = nn.Linear(self.n_in_channels, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        decoder_config = config.transformer.model_copy(update={"cross_attend": True})
        self.decoder = decoder_config.build(dim=config.hidden_dim)

        self.query_embed = nn.Parameter(
            torch.randn(1, config.n_queries, config.hidden_dim)
        )

        self.center_head = nn.Linear(config.hidden_dim, 1)
        self.duration_head = nn.Linear(config.hidden_dim, 1)
        self.start_head = nn.Linear(config.hidden_dim, 1)
        self.end_head = nn.Linear(config.hidden_dim, 1)
        self.class_head = nn.Linear(config.hidden_dim, config.n_classes)

    def _apply_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "center": torch.sigmoid(self.center_head(x)).squeeze(-1),
            "duration": torch.sigmoid(self.duration_head(x)).squeeze(-1),
            "start": torch.sigmoid(self.start_head(x)).squeeze(-1),
            "end": torch.sigmoid(self.end_head(x)).squeeze(-1),
            "class": self.class_head(x),
        }

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.input_proj(memory)
        B, T, D = memory.shape
        pos_enc = _sinusoidal_embeddings(T, D, device=memory.device, dtype=memory.dtype)
        memory_with_pe = memory + pos_enc.unsqueeze(0).expand(B, -1, -1)
        queries = self.query_embed.expand(B, -1, -1)
        output = self.decoder(x=queries, context=memory_with_pe)
        return self._apply_heads(output)
