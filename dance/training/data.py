# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import typing as tp

import neuralset as ns
import pydantic
import torch
from neuralset.dataloader import Batch
from neuralset.events.study import EventsTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import utils
from .extractors import EventEncoder
from .splitters import Splitter, build_splitter

_DETR_RENAMES = {
    "feature_start": "start_target",
    "feature_end": "end_target",
    "feature_class": "class_target",
}


def detr_collate_fn(batches: list[Batch]) -> Batch:
    """Stack single-segment batches into one mini-batch ready for DETR.

    For each tensor key, concatenate across segments. Then rename the
    per-event feature tensors (`feature_start`, `feature_end`,
    `feature_class`) to the names the matcher and losses expect
    (`*_target`) and drop their trailing samples axis so they end up as
    `(batch, max_events)`. The dense target is left as-is.
    """
    if not batches:
        raise ValueError("detr_collate_fn called with empty batch list")

    all_keys: set[str] = set().union(*(b.data.keys() for b in batches))
    stacked: dict[str, torch.Tensor] = {}
    for key in all_keys:
        tensors = [b.data[key] for b in batches if key in b.data]
        try:
            stacked[key] = torch.cat(tensors, dim=0)
        except Exception as exc:
            shapes = [tuple(t.shape) for t in tensors]
            raise RuntimeError(
                f"Failed to stack key {key!r} with shapes {shapes}"
            ) from exc

    for src, dst in _DETR_RENAMES.items():
        if src in stacked:
            stacked[dst] = stacked.pop(src)[:, :, 0]

    return Batch(
        data=stacked,
        segments=[s for b in batches for s in b.segments],
    )


class Data(pydantic.BaseModel):
    """End-to-end data pipeline: from a neuralset Study to train/val/test
    dataloaders.

    Wires together (1) the raw signal extractor (`neuro`, e.g. EegExtractor),
    (2) the per-event feature extractors (`features`, all EventEncoders),
    (3) an optional event-table preprocessor, and (4) a splitter that
    decides which subjects/sessions go to which split. `build()` returns
    a dict {"train": ..., "val": ..., "test": ...} of DataLoaders.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    study: ns.events.Study
    neuro: ns.extractors.BaseExtractor
    features: dict[str, EventEncoder] = pydantic.Field(default_factory=dict)

    preprocessor: EventsTransform | None = None

    splitter_name: str = "kfold"
    splitter_kwargs: dict[str, tp.Any] = pydantic.Field(default_factory=dict)

    duration: float = 8.0
    start: float = 0.0
    training_overlap: float = 0.0

    batch_size: int = 64
    num_workers: int = 0

    use_seizure_sampler: bool = False
    seizure_sampler_alpha: float = 0.01
    seizure_sampler_beta: float = 5.0

    def _splitter(self) -> Splitter:
        # Filter splitter_kwargs to only the params the chosen splitter accepts.
        # defaults.yaml ships kfold's `k_fold/fold_index/valid_seed`, which would
        # leak through ConfDict's deep-merge to e.g. tusz_fixed and crash.
        import inspect

        from .splitters import _SPLITTERS

        cls = _SPLITTERS[self.splitter_name]
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in self.splitter_kwargs.items() if k in accepted}
        return build_splitter(self.splitter_name, **kwargs)

    def _ensure_downloaded(self) -> None:
        """Auto-download the study on first use.

        MOABB studies pull each subject's raw files via mne the first
        time through and write a `timelines.csv` index under
        `study.path`. TUH studies (TUSZ) require manual registration
        with TUH and cannot be auto-fetched; their download raises and
        we expect the user to point `--study-path` at a pre-downloaded
        corpus.
        """
        path = self.study.path
        already = path.exists() and any(path.iterdir())
        if already:
            return
        try:
            self.study.download()
        except NotImplementedError:
            pass

    def _list_segments(self, events, stride: float, drop_incomplete: bool):
        return ns.segments.list_segments(
            events,
            (events.type == self.neuro.event_types),
            start=self.start,
            stride=stride,
            duration=self.duration,
            stride_drop_incomplete=drop_incomplete,
        )

    def build(self) -> dict[str, DataLoader]:
        self._ensure_downloaded()
        events = self.study.run()
        if self.preprocessor is not None:
            events = self.preprocessor.run(events)

        events = ns.events.standardize_events(events)
        self.neuro.prepare(events)
        for feat in self.features.values():
            feat.prepare(events)

        channel_positions = ns.extractors.ChannelPositions(
            neuro=self.neuro,
            allow_missing=True,
        )
        channel_positions.prepare(events)

        extractors = {
            "neuro": self.neuro,
            "channel_positions": channel_positions,
            **self.features,
        }

        splitter = self._splitter()
        train_segments, val_segments, test_segments = splitter(
            events,
            self._list_segments,
            duration=self.duration,
            training_overlap=self.training_overlap,
        )

        loaders: dict[str, DataLoader] = {}
        for split, segments in tqdm(
            [("train", train_segments), ("val", val_segments), ("test", test_segments)],
            desc="building dataloaders",
        ):
            dataset = ns.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,
            )
            sampler = None
            if split == "train" and self.use_seizure_sampler:
                sampler = utils.seizure_weighted_sampler(
                    segments,
                    alpha=self.seizure_sampler_alpha,
                    beta=self.seizure_sampler_beta,
                )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=detr_collate_fn,
                batch_size=self.batch_size,
                shuffle=split == "train" and sampler is None,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        return loaders
