# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import pydantic
import torch
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import ChannelMerger, FourierEmb
from neuraltrain.models.simpleconv import SimpleConv
from neuraltrain.models.simplerconv import SimplerConv
from torch import nn

from .perceiver import Perceiver


def _default_encoder_config() -> SimpleConv:
    """Default conv stack for DANCE: paper-config SimpleConv wrapped with a
    ChannelMerger projecting to 270 virtual channels."""
    return SimpleConv(
        hidden=512,
        depth=10,
        kernel_size=9,
        dilation_growth=2.5,
        initial_linear=256,
        initial_depth=1,
        merger_config=ChannelMerger(
            n_virtual_channels=270,
            fourier_emb_config=FourierEmb(n_freqs=None, total_dim=2048),
            dropout=0.2,
        ),
    )


def _default_perceiver_config() -> Perceiver:
    """Default Perceiver bottleneck for DANCE."""
    return Perceiver(
        num_latents=256,
        depth=6,
        cross_heads=2,
        latent_heads=2,
        cross_dim_head=64,
        latent_dim_head=64,
    )


class Encoder(BaseModelConfig):
    """Brain encoder: convolutional stack followed by a Perceiver bottleneck.

    The conv stack maps raw EEG of shape (batch, channels, time) into a
    sequence of feature vectors at a lower temporal rate. The Perceiver
    then compresses that sequence into a fixed-size set of `num_latents`
    tokens regardless of the input duration. An optional linear projection
    can resize the per-token feature dimension.

    All defaults match the paper configuration; pass an explicit
    `encoder_config` or `perceiver_config` to override.
    """

    dim: int = 128
    encoder_config: SimpleConv | SimplerConv = pydantic.Field(
        default_factory=_default_encoder_config,
    )
    perceiver_config: Perceiver = pydantic.Field(
        default_factory=_default_perceiver_config,
    )
    output_layer_dim: int | None = 0

    def build(self, n_in_channels: int, n_outputs: int | None = None) -> "_Encoder":
        return _Encoder(n_in_channels, n_outputs or self.output_layer_dim, self)


class _Encoder(nn.Module):
    """Conv encoder + Perceiver bottleneck + optional output projection.

    `forward` returns a dict with two tensors of shape
    (batch, num_latents, dim): `z` is the raw Perceiver output (used by
    the dense head) and `c_out` is the same tensor passed through the
    optional output projection (fed to the DETR decoder).
    """

    def __init__(
        self, in_channels: int, out_channels: int | None, config: Encoder
    ) -> None:
        super().__init__()
        self.dim = config.dim
        self.encoder = config.encoder_config.build(in_channels, self.dim)
        self.perceiver = config.perceiver_config.build(self.dim, self.dim)
        self.output_layer = (
            nn.Linear(self.dim, out_channels or self.dim)
            if config.output_layer_dim != 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder.forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )
        z = z.transpose(2, 1)  # (B, F, T) -> (B, T, F)
        z = self.perceiver(z)  # (B, num_latents, F)
        c_out = z if self.output_layer is None else self.output_layer(z)
        return {"z": z, "c_out": c_out}
