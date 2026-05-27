# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
from neuraltrain.models.simpleconv import SimpleConv
from neuraltrain.models.transformer import TransformerEncoder

from dance.models.decoder import Decoder
from dance.models.encoder import Encoder
from dance.models.perceiver import Perceiver


def test_encoder_then_decoder_forward_shapes():
    encoder = Encoder(
        dim=32,
        encoder_config=SimpleConv(
            hidden=32,
            depth=2,
            kernel_size=3,
            dilation_growth=1,
            initial_linear=32,
            initial_depth=1,
            merger_config=None,
        ),
        perceiver_config=Perceiver(
            num_latents=8,
            depth=1,
            latent_dim=32,
            cross_heads=1,
            latent_heads=1,
        ),
        output_layer_dim=None,
    ).build(n_in_channels=4, n_outputs=32)

    decoder = Decoder(
        hidden_dim=64,
        n_queries=5,
        n_classes=3,
        dropout=0.0,
        transformer=TransformerEncoder(heads=2, depth=1),
    ).build(n_in_channels=32)

    eeg = torch.randn(2, 4, 256)
    memory = encoder(eeg)["c_out"]
    out = decoder(memory)

    assert memory.shape == (2, 8, 32)
    assert out["class"].shape == (2, 5, 3)
    assert out["start"].shape == (2, 5)
    assert out["end"].shape == (2, 5)
