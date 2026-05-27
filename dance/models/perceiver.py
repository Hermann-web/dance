# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

from math import log, pi

import torch
import torch.nn as nn
from einops import rearrange, repeat
from neuraltrain.models.base import BaseModelConfig
from perceiver_pytorch import Perceiver as _PtPerceiver


def _fourier_encode(x: torch.Tensor, max_freq: float, num_bands: int = 4) -> torch.Tensor:
    x = x.unsqueeze(-1)
    device, dtype, orig = x.device, x.dtype, x
    scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    return torch.cat((x, orig), dim=-1)


def _sinusoidal_latents(num_latents: int, latent_dim: int) -> torch.Tensor:
    pe = torch.zeros(num_latents, latent_dim)
    position = torch.arange(num_latents, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, latent_dim, 2).float() * (-log(10000.0) / latent_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if latent_dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _Perceiver(_PtPerceiver):
    """Time-locked Perceiver: latents are initialised from sinusoidal
    positional embeddings, so each one starts with a fixed
    "preferred time" anchor rather than a random one.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_latents, latent_dim = self.latents.shape
        with torch.no_grad():
            self.latents.copy_(_sinusoidal_latents(num_latents, latent_dim))

    def forward(self, data, mask=None, return_embeddings=False):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, (
            "input data must have the right number of axes"
        )

        if self.fourier_encode_data:
            axis_pos = [
                torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
                for size in axis
            ]
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)
            enc_pos = _fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
            enc_pos = repeat(enc_pos, "... -> b ...", b=b)
            data = torch.cat((data, enc_pos), dim=-1)

        data = rearrange(data, "b ... d -> b (...) d")
        x = repeat(self.latents, "n d -> b n d", b=b)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        if return_embeddings:
            return x
        return self.to_logits(x)


class Perceiver(BaseModelConfig):
    """Time-locked Perceiver bottleneck. See `_Perceiver` for the latent
    initialisation.
    """

    num_latents: int = 256
    depth: int = 4
    latent_dim: int | None = None
    cross_heads: int = 1
    latent_heads: int = 4
    cross_dim_head: int = 64
    latent_dim_head: int = 64
    input_channels: int | None = None

    def build(self, dim: int, input_channels: int | None = None) -> nn.Module:
        latent_dim = self.latent_dim or dim
        input_channels = input_channels or self.input_channels
        return _Perceiver(
            input_channels=input_channels,
            latent_dim=latent_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            max_freq=10,
            num_freq_bands=6,
            input_axis=1,
            num_classes=1000,
            final_classifier_head=False,
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=1,
        )
