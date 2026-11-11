"""Pretrain the sign model on public data.

    python pretrain.py --data data/public/asl-now-fingerspelling data/public/asl-signs
    python train.py --init runs/pretrained.pt

Each --data folder is one public dataset; which one is told from its contents.
The fingerspelling samples (public_data.py) are still hands with no body, so
they teach hand shapes only. The isolated signs (asl_signs.py) have movement,
both hands and the body, but they are ASL signs. Training on recorded clips
afterwards (train.py --init) is what makes the model fit the signs as they are
made here; the checkpoint is never served.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

import asl_signs
import public_data
from checkpoint import save_checkpoint
from data import Clip
from train import SignNet, probabilities, training_batch

from app.recognition.features import IDLE_LABEL  # path set up by data.py

HELD_OUT = 0.15
BATCH = 64
# Clips augmented at a time. A whole epoch of the isolated signs at once would not fit in memory.
CHUNK = BATCH * 64


def load(folders: list[Path], per_sign: int | None) -> tuple[list[Clip], str]:
    """The clips of every dataset folder, and the names of the datasets they came from."""
    clips, sources = [], []
    for folder in folders:
        if (folder / "train.csv").is_file():
            found, source = asl_signs.load_sequences(folder, per_sign), asl_signs.SOURCE
        else:
            found, source = public_data.load_samples(folder), public_data.SOURCE
        if found:
            clips += found
            sources.append(source)
    return clips, " + ".join(dict.fromkeys(sources))


def split(clips: list[Clip], rng: np.random.Generator) -> tuple[list[Clip], list[Clip], str]:
    """Hold out whole signers where the data names them, and a share of every label where it does not.

    The fingerspelling dataset does not say who signed what, so the same people
    are on both sides of its split and accuracy on it is optimistic.
    """
    named = sorted({c.signer for c in clips} - {public_data.SIGNER})
    held = set()
    if len(named) > 1:
        held = set(rng.choice(named, size=max(1, round(len(named) * HELD_OUT)), replace=False).tolist())
    train = [c for c in clips if c.signer in named and c.signer not in held]
    test = [c for c in clips if c.signer in held]

    anonymous = [c for c in clips if c.signer not in named]
    for label in sorted({c.label for c in anonymous}):
        group = [c for c in anonymous if c.label == label]
        order = rng.permutation(len(group))
        cut = int(len(group) * HELD_OUT)
        test += [group[i] for i in order[:cut]]
        train += [group[i] for i in order[cut:]]

    notes = []
    if held:
        notes.append(f"{len(held)} of {len(named)} signers held out")
    if anonymous:
        notes.append("same people in both halves for the unnamed data: optimistic")
    return train, test, "; ".join(notes)


def _chunks(count: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Indices 0..count-1 in near-equal chunks, shuffled when given a generator."""
    order = rng.permutation(count) if rng is not None else np.arange(count)
    return np.array_split(order, -(-count // CHUNK))


def _batches(count: int) -> int:
    return -(-count // BATCH)


def top1(model: nn.Module, clips: list[Clip], labels: list[str]) -> float:
    correct = 0
    for chunk in _chunks(len(clips)):
        group = [clips[i] for i in chunk]
        truth = np.array([labels.index(c.label) for c in group])
        correct += int((probabilities(model, group).argmax(axis=1) == truth).sum())
    return correct / len(clips)


def pretrain(data: Path | list[Path], out: Path, epochs: int, seed: int, per_sign: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    folders = [data] if isinstance(data, Path) else list(data)
    clips, source = load(folders, per_sign)
    if len({c.label for c in clips}) < 2:
        where = ", ".join(str(f) for f in folders)
        raise SystemExit(
            f"Found {len(clips)} usable samples under {where}; see public_data.py and asl_signs.py for where to get them."
        )
    train_clips, test_clips, how = split(clips, rng)
    # Same label set shape as train.py: straddling windows are labelled idle.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    print(f"{len(train_clips)} training samples, {len(test_clips)} held out, {len(labels) - 1} labels ({source})")

    model = SignNet(len(labels))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = sum(_batches(len(chunk)) for chunk in _chunks(len(train_clips)))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for chunk in _chunks(len(train_clips), rng):
            x, y = training_batch([train_clips[i] for i in chunk], labels, rng)
            model.train()
            for batch in torch.arange(len(chunk)).tensor_split(_batches(len(chunk))):
                optimiser.zero_grad()
                loss = loss_fn(model(x[batch]), y[batch])
                loss.backward()
                optimiser.step()
                schedule.step()
                total += loss.item() * len(batch)
        if epoch % 5 == 0 or epoch == epochs or len(train_clips) > CHUNK:
            print(f"epoch {epoch:3d}  loss {total / len(train_clips):.3f}")

    report = {"samples": len(clips), "labels": labels, "epochs": epochs, "seed": seed, "source": source}
    # A held-out signer may have made a sign nobody in training did; it cannot be scored.
    scored = [c for c in test_clips if c.label in labels]
    if scored:
        report |= {"test_samples": len(scored), "top1": top1(model, scored, labels), "held_out": how}
        print(f"held-out top-1 {report['top1']:.1%} on {len(scored)} samples ({how})")
    save_checkpoint(model, labels, source, out)
    print(f"checkpoint: {out}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", default=[here / "data" / "public" / "asl-now-fingerspelling"],
                        help="one or more public dataset folders")
    parser.add_argument("--out", type=Path, default=here / "runs" / "pretrained.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-sign", type=int, default=100,
                        help="isolated-sign clips to use for each sign; 0 uses all of them (about 94,000)")
    args = parser.parse_args()
    pretrain(args.data, args.out, args.epochs, args.seed, args.per_sign or None)
