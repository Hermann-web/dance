# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import torch
from neuraltrain.models.common import ChannelMerger, FourierEmb
from neuraltrain.models.simpleconv import SimpleConv
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

from dance import Dance
from dance.losses import ConsistencyLoss, DenseLoss, DetrLoss, IoULoss
from dance.matcher import HungarianMatcher
from dance.models.decoder import Decoder
from dance.models.encoder import Encoder
from dance.models.perceiver import Perceiver


# A subclass that rebuilds Dance with a tiny architecture, so CPU tests
# stay sub-second. Demonstrates the documented "subclass to tweak"
# escape hatch from the Dance docstring.
class _TinyDance(Dance):
    _NUM_LATENTS = 8
    _DIM = 16

    def __init__(
        self, *, n_channels, n_classes, n_queries, duration, use_channel_merger=True
    ):
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.duration = duration
        self.use_channel_merger = use_channel_merger
        self.num_latents = self._NUM_LATENTS
        self.frequency = self._NUM_LATENTS / duration

        merger = (
            ChannelMerger(
                n_virtual_channels=8,
                fourier_emb_config=FourierEmb(n_freqs=None, total_dim=32),
            )
            if use_channel_merger
            else None
        )
        self.encoder = Encoder(
            dim=self._DIM,
            encoder_config=SimpleConv(
                hidden=16,
                depth=2,
                kernel_size=3,
                dilation_growth=1,
                initial_linear=16,
                initial_depth=1,
                merger_config=merger,
            ),
            perceiver_config=Perceiver(
                num_latents=self._NUM_LATENTS,
                depth=1,
                cross_heads=1,
                latent_heads=1,
            ),
            output_layer_dim=0,
        ).build(n_in_channels=n_channels, n_outputs=self._DIM)
        # x-transformers requires dim_head >= 32; with 2 heads -> hidden >= 64
        self.decoder = Decoder(
            hidden_dim=64,
            n_queries=n_queries,
            transformer=TransformerEncoder(depth=1, heads=2),
            n_classes=n_classes,
        ).build(n_in_channels=self._DIM)
        self.dense_head = nn.Linear(self._DIM, n_classes)

        matcher = HungarianMatcher(
            weight_class=1.0, weight_iou=5.0, window_length=duration
        )
        self.detr_loss = DetrLoss(
            matcher,
            nn.CrossEntropyLoss(),
            IoULoss(mode="start_end").build(),
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


def _tiny_batch(B=2, n_channels=4, T=64, num_latents=8, with_positions=True):
    batch = {
        "eeg": torch.randn(B, n_channels, T),
        "start": torch.tensor([[0.1, 0.5, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0]]),
        "end": torch.tensor([[0.3, 0.7, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0, 0.0]]),
        "class": torch.tensor([[1, 2, 0, 0, 0], [1, 0, 0, 0, 0]]),
        "dense": torch.zeros(B, num_latents, dtype=torch.long),
    }
    if with_positions:
        batch["channel_positions"] = torch.rand(B, n_channels, 2)
    return batch


def _tiny_model(**kw):
    cfg = dict(n_channels=4, n_classes=3, n_queries=5, duration=1.0)
    cfg.update(kw)
    return _TinyDance(**cfg)


def test_dance_forward_returns_loss_and_predictions():
    """With targets in the batch: scalar loss + four prediction tensors."""
    model = _tiny_model()
    out = model(_tiny_batch())
    assert out["loss"].ndim == 0
    assert out["pred_class"].shape == (2, 5, 3)
    assert out["pred_start"].shape == (2, 5)
    assert out["pred_end"].shape == (2, 5)
    assert out["pred_dense"].shape == (2, 8, 3)
    out["loss"].backward()
    assert {"detr_class", "detr_iou", "dense", "consistency"} <= out[
        "loss_details"
    ].keys()


def test_dance_forward_inference_only_when_no_targets():
    """Without targets the model returns predictions only — no loss key."""
    model = _tiny_model()
    out = model({"eeg": torch.randn(2, 4, 64), "channel_positions": torch.rand(2, 4, 2)})
    assert "loss" not in out
    assert "loss_details" not in out
    assert out["pred_class"].shape == (2, 5, 3)


def test_dance_with_merger_is_channel_agnostic():
    """The ChannelMerger lets the same model handle different channel counts."""
    model = _tiny_model(n_channels=4)
    for actual in (4, 16, 32):
        out = model(
            {
                "eeg": torch.randn(1, actual, 64),
                "channel_positions": torch.rand(1, actual, 2),
            }
        )
        assert out["pred_class"].shape == (1, 5, 3)


def test_dance_derives_dense_from_events_when_missing():
    """If `dense` is absent, it is rendered from (start, end, class)."""
    model = _tiny_model()
    batch = _tiny_batch()
    batch.pop("dense")
    out = model(batch)
    assert out["loss"].ndim == 0
    out["loss"].backward()


def test_dense_from_events_paints_spans_and_ignores_padding():
    """T=8 with two events item-0 and one event + padding item-1."""
    model = _tiny_model()  # _NUM_LATENTS = 8
    # item 0: paint [0, 4)=1 then [4, 8)=2
    # item 1: paint [0, 2)=3 then padding (class=0) then zero-length (s==e)
    start = torch.tensor([[0.00, 0.50, 0.0, 0.0], [0.00, 0.50, 0.50, 0.0]])
    end = torch.tensor([[0.50, 1.00, 0.0, 0.0], [0.25, 0.75, 0.50, 0.0]])
    cls = torch.tensor([[1, 2, 0, 0], [3, 0, 1, 0]])
    dense = model._dense_from_events(start, end, cls)
    assert dense.shape == (2, 1, 8)
    assert dense.dtype == torch.long
    expected = torch.tensor(
        [
            [[1, 1, 1, 1, 2, 2, 2, 2]],
            [[3, 3, 0, 0, 0, 0, 0, 0]],
        ]
    )
    assert torch.equal(dense, expected)


def test_dense_from_events_overlap_last_event_wins():
    """Documented policy: overlapping spans, later index in `class` wins."""
    model = _tiny_model()
    start = torch.tensor([[0.000, 0.250, 0.0, 0.0]])
    end = torch.tensor([[0.750, 1.000, 0.0, 0.0]])
    cls = torch.tensor([[1, 2, 0, 0]])
    dense = model._dense_from_events(start, end, cls)
    expected = torch.tensor([[[1, 1, 2, 2, 2, 2, 2, 2]]])
    assert torch.equal(dense, expected)


def test_dance_without_merger_uses_raw_channels():
    """`use_channel_merger=False` skips the merger; no `channel_positions` needed."""
    model = _tiny_model(use_channel_merger=False)
    out = model(_tiny_batch(with_positions=False))
    assert out["pred_class"].shape == (2, 5, 3)
    out["loss"].backward()


def test_paper_dance_instantiates():
    """End-to-end smoke for the paper config (28.8M params).
    Requires neuraltrain >= 0.2.2 (earlier versions rejected the float
    `dilation_growth=2.5` paper hyperparameter at SimpleConv validation)."""
    Dance(n_channels=16, n_classes=3, n_queries=150, duration=32.0)
