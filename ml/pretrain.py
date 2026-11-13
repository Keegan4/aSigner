"""Pretrain the sign model on public data: ASLNow! fingerspelling, plus CASL-W60 when present.

    python pretrain.py --data data/public/asl-now-fingerspelling --casl data/public/casl-w60 --out runs/pretrained.pt
    python train.py --init runs/pretrained.pt

ASLNow! samples are still hands with no body; CASL-W60 adds movement and pose
from another sign language (see public_data.py). Loss and macro-F1 for every
epoch go into runs/pretrained.json next to the checkpoint (plot_pretrain.py
draws them). Training on recorded clips afterwards (train.py --init) is what makes the
model fit the signs as they are made here; the checkpoint is never served.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from checkpoint import save_checkpoint
from data import Clip
from public_data import CASL_SOURCE, SOURCE, load_casl, load_samples
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


def macro_f1(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Mean F1 over the classes present in `truth`."""
    scores = []
    for k in np.unique(truth):
        hits = np.sum((predicted == k) & (truth == k))
        wrong = np.sum((predicted == k) & (truth != k)) + np.sum((predicted != k) & (truth == k))
        scores.append(2 * hits / (2 * hits + wrong) if hits else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate(model: nn.Module, clips: list[Clip], labels: list[str], loss_fn: nn.Module) -> dict:
    truth = np.array([labels.index(c.label) for c in clips])
    probs = probabilities(model, clips)
    log_probs = torch.from_numpy(np.log(np.clip(probs, 1e-12, None)))
    return {
        "loss": float(loss_fn(log_probs, torch.from_numpy(truth)).item()),
        "f1": macro_f1(probs.argmax(axis=1), truth),
        "top1": float((probs.argmax(axis=1) == truth).mean()),
        "samples": len(clips),
    }


def pretrain(data: Path, out: Path, epochs: int, seed: int, casl: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    clips = load_samples(data)
    if len({c.label for c in clips}) < 2:
        raise SystemExit(f"Found {len(clips)} usable samples under {data}; see public_data.py for where to get them.")
    train_clips, test_clips = split(clips, rng)
    held_out = {"asl-now": test_clips}
    sources = [SOURCE]
    casl_train, casl_test = load_casl(casl) if casl else ([], [])
    if casl_train:
        train_clips += casl_train
        held_out["casl-w60"] = casl_test
        sources.append(CASL_SOURCE)
        print(f"CASL-W60: {len(casl_train)} training clips, {len(casl_test)} test clips")
    elif casl:
        print(f"no CASL-W60 parquet files under {casl}; using ASLNow! only")
    source = " + ".join(sources)
    # Same label set shape as train.py: straddling windows are labelled idle.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    held_out = {name: [c for c in group if c.label in labels] for name, group in held_out.items() if group}
    print(f"{len(train_clips)} training clips, {sum(map(len, held_out.values()))} held out, "
          f"{len(labels) - 1} labels ({source})")

    device = pick_device()
    model = SignNet(len(labels)).to(device)
    print(f"training on {device}")
    steps = (len(train_clips) + 63) // 64
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, epochs=epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        x_all, y_all = training_batch(train_clips, labels, rng)
        x_all, y_all = x_all.to(device), y_all.to(device)
        model.train()
        total, predicted = 0.0, np.empty(len(train_clips), dtype=np.int64)
        for batch in torch.from_numpy(rng.permutation(len(train_clips))).tensor_split(steps):
            batch = batch.to(device)
            optimiser.zero_grad()
            logits = model(x_all[batch])
            loss = loss_fn(logits, y_all[batch])
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
            predicted[batch.cpu().numpy()] = logits.argmax(dim=1).cpu().numpy()
        # Training figures are on that epoch's augmented clips, with dropout on.
        row = {"epoch": epoch, "train": {"loss": total / len(train_clips),
                                         "f1": macro_f1(predicted, y_all.cpu().numpy())}}
        row |= {name: evaluate(model, group, labels, loss_fn) for name, group in held_out.items()}
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['train']['loss']:.3f}  "
              + "  ".join(f"{name} f1 {row[name]['f1']:.3f}" for name in held_out))

    report = {"samples": len(clips) + len(casl_train) + len(casl_test), "labels": labels, "epochs": epochs,
              "seed": seed, "source": source, "history": history}
    if "asl-now" in held_out:
        report |= {"test_samples": len(held_out["asl-now"]), "top1": history[-1]["asl-now"]["top1"]}
        print(f"held-out ASLNow! top-1 {report['top1']:.1%} on {report['test_samples']} samples "
              "(same people in both halves: optimistic)")
    if "casl-w60" in held_out:
        report["casl_top1"] = history[-1]["casl-w60"]["top1"]
        print(f"CASL-W60 test top-1 {report['casl_top1']:.1%} on {len(held_out['casl-w60'])} clips")
    save_checkpoint(model, labels, source, out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"checkpoint: {out}\nhistory: {out.with_suffix('.json')}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=here / "data" / "public" / "asl-now-fingerspelling")
    parser.add_argument("--out", type=Path, default=here / "runs" / "pretrained.pt")
    parser.add_argument("--casl", type=Path, default=here / "data" / "public" / "casl-w60",
                        help="CASL-W60 parquet files; used when present")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pretrain(args.data, args.out, args.epochs, args.seed, args.casl)
