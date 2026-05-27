"""Train DANCE on BI2014b from scratch + MOABB + torch.

Usage:
    python train.py
"""

from __future__ import annotations

import torch
from data import (
    DURATION_S,
    MAX_EVENTS,
    N_CLASSES,
    DanceBI2014bDataset,
    collate,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MultilabelF1Score

from dance import Dance
from dance.metrics import F1Event

SUBJECTS = list(range(1, 39))  # all 38 BI2014b subjects
N_EPOCHS = 50
BATCH_SIZE = 8
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_loaders():
    ds = DanceBI2014bDataset(SUBJECTS)
    idx = list(range(len(ds)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        Subset(ds, test_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    return train_loader, test_loader


def _move(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: Dance, loader: DataLoader) -> dict[str, float]:
    model.eval()
    from dance.utils import (
        events_to_mask,
        extract_events_from_detr_batch,
    )

    f1_sample = MultilabelF1Score(num_labels=N_CLASSES, average="macro").to(DEVICE)
    f1_event = F1Event(iou_threshold=0.5)
    for batch in loader:
        batch = _move(batch, DEVICE)
        out = model(batch)
        preds = {
            "class": out["pred_class"],
            "start": out["pred_start"],
            "end": out["pred_end"],
        }
        targets = {"class": batch["class"], "start": batch["start"], "end": batch["end"]}
        pred_events, gt_events = extract_events_from_detr_batch(
            preds, targets, window_length=DURATION_S
        )
        # F1-sample: render predictions + targets into per-token class masks.
        for pe, ge in zip(pred_events, gt_events):
            pred_mask = events_to_mask(pe, DURATION_S, model.frequency, N_CLASSES)
            gt_mask = events_to_mask(ge, DURATION_S, model.frequency, N_CLASSES)
            f1_sample.update(
                pred_mask.to(DEVICE).unsqueeze(0), gt_mask.to(DEVICE).unsqueeze(0)
            )
        f1_event.update(pred_events, gt_events)
    return {
        "f1_sample": float(f1_sample.compute()),
        "f1_event": float(f1_event.compute()),
    }


def main() -> None:
    train_loader, test_loader = _make_loaders()
    print(f"train windows: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}")

    n_channels = next(iter(train_loader))["eeg"].shape[1]
    model = Dance(
        n_channels=n_channels,
        n_classes=N_CLASSES,
        n_queries=MAX_EVENTS,
        duration=DURATION_S,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    for epoch in range(N_EPOCHS):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            batch = _move(batch, DEVICE)
            out = model(batch)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            loss_total += out["loss"].item()
        scores = evaluate(model, test_loader) if epoch % 5 == 0 else {}
        msg = f"epoch {epoch:3d}  train_loss={loss_total / len(train_loader):.3f}"
        if scores:
            msg += f"  test_f1_sample={scores['f1_sample']:.3f}  test_f1_event={scores['f1_event']:.3f}"
        print(msg, flush=True)

    final = evaluate(model, test_loader)
    print(
        f"\nFINAL  test_f1_sample={final['f1_sample']:.3f}  test_f1_event={final['f1_event']:.3f}"
    )


if __name__ == "__main__":
    main()
