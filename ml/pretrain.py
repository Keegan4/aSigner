"""Pretrain the sign model on public data: ASLNow! fingerspelling, plus CASL-W60, WLASL and asl-signs when present.

    python pretrain.py --data data/public/asl-now-fingerspelling --casl data/public/casl-w60 \\
        --wlasl data/public/wlasl-landmarks --asl-signs data/public/asl-signs --out runs/pretrained.pt
    python train.py --init runs/pretrained.pt

ASLNow! samples are still hands with no body; CASL-W60 and WLASL add movement
and pose (see public_data.py), and so does Kaggle's much larger asl-signs
(see asl_signs.py). Loss and macro-F1 for every
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

from asl_signs import SOURCE as ASL_SIGNS_SOURCE
from asl_signs import load_asl_signs
from checkpoint import save_checkpoint
from data import Clip
from public_data import CASL_SOURCE, SOURCE, WLASL_SOURCE, load_casl, load_samples, load_wlasl
from train import SignNet, model_inputs, pick_device, probabilities, training_batch

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


def evaluate(model: nn.Module, clips: list[Clip], labels: list[str], loss_fn: nn.Module,
             inputs: np.ndarray | None = None) -> dict:
    """Score held-out clips. `inputs` are their `model_inputs`, when already worked out."""
    truth = np.array([labels.index(c.label) for c in clips])
    probs = probabilities(model, clips if inputs is None else inputs)
    log_probs = torch.from_numpy(np.log(np.clip(probs, 1e-12, None)))
    return {
        "loss": float(loss_fn(log_probs, torch.from_numpy(truth)).item()),
        "f1": macro_f1(probs.argmax(axis=1), truth),
        "top1": float((probs.argmax(axis=1) == truth).mean()),
        "samples": len(clips),
    }


def pretrain(data: Path, out: Path, epochs: int, seed: int, casl: Path | None = None,
             wlasl: Path | None = None, warmup: int | None = None, asl_signs: Path | None = None,
             per_sign: int | None = 100) -> dict:
    """`warmup` is how many epochs the learning rate climbs for before it starts
    to fall; left out, it is the first 30% of them. `per_sign` limits how many
    asl-signs clips of each sign are read (None for all, about 380 a sign)."""
    if warmup is not None and not 0 < warmup < epochs:
        raise SystemExit(f"--warmup must be between 1 and {epochs - 1} for {epochs} epochs.")
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
        print(f"no CASL-W60 parquet files under {casl}; leaving it out")
    wlasl_train, wlasl_test = load_wlasl(wlasl) if wlasl else ([], [])
    if wlasl_train:
        train_clips += wlasl_train
        held_out["wlasl"] = wlasl_test
        sources.append(WLASL_SOURCE)
        print(f"WLASL: {len(wlasl_train)} training clips, {len(wlasl_test)} held out")
    elif wlasl:
        print(f"no WLASL parquet files under {wlasl}; leaving it out")
    asl_train, asl_test = load_asl_signs(asl_signs, per_sign) if asl_signs else ([], [])
    if asl_train:
        train_clips += asl_train
        held_out["asl-signs"] = asl_test
        sources.append(ASL_SIGNS_SOURCE)
        print(f"asl-signs: {len(asl_train)} training clips, {len(asl_test)} held out "
              f"({len({c.signer for c in asl_test})} whole signers)")
    elif asl_signs:
        print(f"no asl-signs train.csv under {asl_signs}; leaving it out")
    source = " + ".join(sources)
    # Same label set shape as train.py: windows split evenly between two signs are labelled idle.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    held_out = {name: [c for c in group if c.label in labels] for name, group in held_out.items() if group}
    print(f"{len(train_clips)} training clips, {sum(map(len, held_out.values()))} held out, "
          f"{len(labels) - 1} labels ({source})")

    device = pick_device()
    model = SignNet(len(labels)).to(device)
    print(f"training on {device}")
    steps = (len(train_clips) + 63) // 64
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, epochs=epochs, steps_per_epoch=steps,
                                                   pct_start=warmup / epochs if warmup else 0.3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Held-out clips are not augmented, so their inputs are the same every epoch.
    held_out_inputs = {name: model_inputs(group) for name, group in held_out.items()}
    history = []
    for epoch in range(1, epochs + 1):
        # The epoch's inputs stay in main memory and go to the device a batch at
        # a time: with asl-signs they are too big to sit on a small GPU whole.
        x_all, y_all = training_batch(train_clips, labels, rng)
        model.train()
        total, predicted = 0.0, np.empty(len(train_clips), dtype=np.int64)
        for batch in torch.from_numpy(rng.permutation(len(train_clips))).tensor_split(steps):
            optimiser.zero_grad()
            logits = model(x_all[batch].to(device))
            loss = loss_fn(logits, y_all[batch].to(device))
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
            predicted[batch.numpy()] = logits.argmax(dim=1).cpu().numpy()
        # Training figures are on that epoch's augmented clips, with dropout on.
        row = {"epoch": epoch, "train": {"loss": total / len(train_clips),
                                         "f1": macro_f1(predicted, y_all.numpy())}}
        row |= {name: evaluate(model, group, labels, loss_fn, held_out_inputs[name])
                for name, group in held_out.items()}
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['train']['loss']:.3f}  "
              + "  ".join(f"{name} f1 {row[name]['f1']:.3f}" for name in held_out))

    report = {"samples": (len(clips) + len(casl_train) + len(casl_test) + len(wlasl_train) + len(wlasl_test)
                          + len(asl_train) + len(asl_test)),
              "labels": labels, "epochs": epochs, "warmup_epochs": warmup if warmup else 0.3 * epochs,
              "seed": seed, "source": source, "history": history}
    if "asl-now" in held_out:
        report |= {"test_samples": len(held_out["asl-now"]), "top1": history[-1]["asl-now"]["top1"]}
        print(f"held-out ASLNow! top-1 {report['top1']:.1%} on {report['test_samples']} samples "
              "(same people in both halves: optimistic)")
    if "casl-w60" in held_out:
        report["casl_top1"] = history[-1]["casl-w60"]["top1"]
        print(f"CASL-W60 test top-1 {report['casl_top1']:.1%} on {len(held_out['casl-w60'])} clips")
    if "wlasl" in held_out:
        report["wlasl_top1"] = history[-1]["wlasl"]["top1"]
        print(f"WLASL held-out top-1 {report['wlasl_top1']:.1%} on {len(held_out['wlasl'])} clips")
    if "asl-signs" in held_out:
        report["asl_signs_top1"] = history[-1]["asl-signs"]["top1"]
        print(f"asl-signs held-out top-1 {report['asl_signs_top1']:.1%} on {len(held_out['asl-signs'])} clips "
              "(signers the model never saw)")
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
    parser.add_argument("--wlasl", type=Path, default=here / "data" / "public" / "wlasl-landmarks",
                        help="WLASL landmark parquet files (academic, non-commercial); used when present")
    parser.add_argument("--asl-signs", type=Path, default=here / "data" / "public" / "asl-signs",
                        help="Kaggle's asl-signs folder (the one holding train.csv); used when present")
    parser.add_argument("--per-sign", type=int, default=100,
                        help="asl-signs clips to read for each sign (default 100; 0 for all of them)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup", type=int,
                        help="epochs the learning rate rises for before it falls (default: 30%% of --epochs)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pretrain(args.data, args.out, args.epochs, args.seed, args.casl, args.wlasl, args.warmup,
             args.asl_signs, args.per_sign or None)
