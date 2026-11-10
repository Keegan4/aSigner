"""Train the sign classifier and export it for the server.

    python train.py --clips ../server/data/clips --out ../server/models/signs.onnx

Writes signs.onnx and signs.labels.json, plus report.json and confusion.csv
under runs/<time>/.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import Clip, augment, load_clips, to_model_input

from app.features import CLIP_FRAMES, NORMALISED_SIZE  # path set up by data.py
from app.onnx_recogniser import OnnxRecogniser


class SignNet(nn.Module):
    """A small temporal convolution network over normalised landmark frames."""

    def __init__(self, classes: int, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(NORMALISED_SIZE, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width, classes))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:  # (batch, time, features)
        return self.head(self.body(clip.transpose(1, 2)).mean(dim=2))


def split(clips: list[Clip], test_signer: str | None, rng: np.random.Generator):
    """Hold out a whole signer when there is more than one; otherwise 20% of each sign."""
    signers = Counter(c.signer for c in clips)
    if len(signers) > 1:
        held = test_signer or min(signers, key=signers.get)
        train = [c for c in clips if c.signer != held]
        test = [c for c in clips if c.signer == held]
        return train, test, f"signer {held} held out"
    train, test = [], []
    for label in sorted({c.label for c in clips}):
        group = [c for c in clips if c.label == label]
        order = rng.permutation(len(group))
        cut = max(1, len(group) // 5) if len(group) > 1 else 0
        test += [group[i] for i in order[:cut]]
        train += [group[i] for i in order[cut:]]
    return train, test, "same signer in train and test: accuracy is optimistic"


def evaluate(model: nn.Module, clips: list[Clip], labels: list[str]):
    x = torch.from_numpy(np.stack([to_model_input(c.frames) for c in clips]))
    y = np.array([labels.index(c.label) for c in clips])
    model.eval()
    with torch.no_grad():
        logits = model(x).numpy()
    ranked = np.argsort(-logits, axis=1)
    top1 = float((ranked[:, 0] == y).mean())
    top5 = float((ranked[:, :5] == y[:, None]).any(axis=1).mean())
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, guess in zip(y, ranked[:, 0]):
        confusion[truth, guess] += 1
    return top1, top5, confusion


def train(clips_dir: Path, out: Path, epochs: int, seed: int, test_signer: str | None, runs: Path):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    clips = load_clips(clips_dir)
    labels = sorted({c.label for c in clips})
    if len(labels) < 2:
        raise SystemExit(f"Need clips for at least two signs; found {len(clips)} clips, {len(labels)} signs.")
    train_clips, test_clips, split_note = split(clips, test_signer, rng)
    labels = sorted({c.label for c in train_clips})
    test_clips = [c for c in test_clips if c.label in labels]
    print(f"{len(train_clips)} training clips, {len(test_clips)} test clips, {len(labels)} signs ({split_note})")

    model = SignNet(len(labels))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    y_all = torch.tensor([labels.index(c.label) for c in train_clips])

    for epoch in range(1, epochs + 1):
        model.train()
        # Fresh augmentation every epoch, so the model never sees the same clip twice.
        x_all = torch.from_numpy(np.stack([to_model_input(augment(c.frames, rng)) for c in train_clips]))
        order = torch.from_numpy(rng.permutation(len(train_clips)))
        total = 0.0
        for batch in order.split(64):
            optimiser.zero_grad()
            loss = loss_fn(model(x_all[batch]), y_all[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        schedule.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}  loss {total / len(train_clips):.3f}")

    report = {"clips": len(clips), "signs": labels, "split": split_note, "epochs": epochs, "seed": seed}
    confusion = None
    if test_clips:
        top1, top5, confusion = evaluate(model, test_clips, labels)
        report |= {"test_clips": len(test_clips), "top1": top1, "top5": top5}
        print(f"test top-1 {top1:.1%}  top-5 {top5:.1%}  ({split_note})")

    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, CLIP_FRAMES, NORMALISED_SIZE)
    torch.onnx.export(model, (example,), str(out), input_names=["clip"], output_names=["logits"],
                      dynamo=True)
    out.with_suffix(".labels.json").write_text(json.dumps(labels), encoding="utf-8")

    # The exported file must give the server the same answer PyTorch gives here.
    probe = to_model_input(clips[0].frames)
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(probe)[None]), dim=1)[0].numpy()
    exported = OnnxRecogniser(out).predict(probe)
    if not np.allclose(expected, exported, atol=1e-4):
        raise SystemExit("Exported model disagrees with the trained model.")

    run = runs / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if confusion is not None:
        with open(run / "confusion.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["truth \\ guess", *labels])
            writer.writerows([label, *row] for label, row in zip(labels, confusion.tolist()))
    print(f"model: {out}\nreport: {run}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=here.parent / "server" / "data" / "clips")
    parser.add_argument("--out", type=Path, default=here.parent / "server" / "models" / "signs.onnx")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-signer", help="signer id to hold out (default: the one with fewest clips)")
    parser.add_argument("--runs", type=Path, default=here / "runs")
    args = parser.parse_args()
    train(args.clips, args.out, args.epochs, args.seed, args.test_signer, args.runs)
