"""Pretrain the sign model on public fingerspelling data.

    python pretrain.py --data data/public/asl-now-fingerspelling --out runs/pretrained.pt
    python train.py --init runs/pretrained.pt

The public samples are still hands with no body, so this teaches hand shapes
only. Training on recorded clips afterwards (train.py --init) is what makes the
model fit the signs as they are made here; the checkpoint is never served.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from checkpoint import save_checkpoint
from data import Clip
from public_data import SOURCE, load_samples
from train import SignNet, pick_device, probabilities, training_batch

from app.recognition.features import IDLE_LABEL  # path set up by data.py

HELD_OUT = 0.15


def split(clips: list[Clip], rng: np.random.Generator) -> tuple[list[Clip], list[Clip]]:
    """Hold out a share of every letter. The dataset does not say who signed what,
    so the same people are on both sides and the accuracy is optimistic."""
    train, test = [], []
    for label in sorted({c.label for c in clips}):
        group = [c for c in clips if c.label == label]
        order = rng.permutation(len(group))
        cut = int(len(group) * HELD_OUT)
        test += [group[i] for i in order[:cut]]
        train += [group[i] for i in order[cut:]]
    return train, test


def pretrain(data: Path, out: Path, epochs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    clips = load_samples(data)
    if len({c.label for c in clips}) < 2:
        raise SystemExit(f"Found {len(clips)} usable samples under {data}; see public_data.py for where to get them.")
    train_clips, test_clips = split(clips, rng)
    # Same label set shape as train.py: straddling windows are labelled idle.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    print(f"{len(train_clips)} training samples, {len(test_clips)} held out, {len(labels) - 1} letters ({SOURCE})")

    device = pick_device()
    model = SignNet(len(labels)).to(device)
    print(f"training on {device}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3, epochs=epochs, steps_per_epoch=(len(train_clips) + 63) // 64
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, epochs + 1):
        x_all, y_all = training_batch(train_clips, labels, rng)
        x_all, y_all = x_all.to(device), y_all.to(device)
        model.train()
        total = 0.0
        batches = torch.from_numpy(rng.permutation(len(train_clips))).tensor_split((len(train_clips) + 63) // 64)
        for batch in batches:
            optimiser.zero_grad()
            loss = loss_fn(model(x_all[batch]), y_all[batch])
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}  loss {total / len(train_clips):.3f}")

    report = {"samples": len(clips), "labels": labels, "epochs": epochs, "seed": seed, "source": SOURCE}
    if test_clips:
        truth = np.array([labels.index(c.label) for c in test_clips])
        top1 = float((probabilities(model, test_clips).argmax(axis=1) == truth).mean())
        report |= {"test_samples": len(test_clips), "top1": top1}
        print(f"held-out top-1 {top1:.1%} on {len(test_clips)} samples (same people in both halves: optimistic)")
    save_checkpoint(model, labels, SOURCE, out)
    print(f"checkpoint: {out}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=here / "data" / "public" / "asl-now-fingerspelling")
    parser.add_argument("--out", type=Path, default=here / "runs" / "pretrained.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pretrain(args.data, args.out, args.epochs, args.seed)
