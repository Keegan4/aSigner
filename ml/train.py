"""Train the sign classifier and export it for the server.

    python train.py --clips ../server/data/clips --out ../server/models/signs.onnx

Writes signs.onnx (labels and pass score inside it), plus report.json and
confusion.csv under runs/<time>/.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from data import Clip, augment, load_clips, to_model_input, transition

from app.recognition.features import CLIP_FRAMES, IDLE_LABEL, NORMALISED_SIZE  # path set up by data.py
from app.recognition.onnx_recogniser import LABELS_KEY, PASS_SCORE_KEY, OnnxRecogniser

TRANSITION_SHARE = 0.3  # of training clips shown as a window straddling two signs
SIGN_FRACTION = 0.6  # a straddling window counts as the sign only above this much of it
MIN_CLIPS_FOR_THRESHOLD = 30


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


def training_batch(clips: list[Clip], labels: list[str], rng: np.random.Generator):
    """One epoch's inputs: every clip freshly augmented, some as straddling windows."""
    xs, ys = [], []
    for clip in clips:
        frames, label = clip.frames, clip.label
        if label != IDLE_LABEL and len(clips) > 1 and rng.random() < TRANSITION_SHARE:
            other = clips[rng.integers(len(clips))]
            if other.label != label:
                frames, fraction = transition(frames, other.frames, rng)
                if fraction < SIGN_FRACTION:
                    label = IDLE_LABEL
        xs.append(to_model_input(augment(frames, rng)))
        ys.append(labels.index(label))
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


def probabilities(model: nn.Module, clips: list[Clip]) -> np.ndarray:
    x = torch.from_numpy(np.stack([to_model_input(c.frames) for c in clips]))
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).numpy()


def threshold_curve(probs: np.ndarray, truth: np.ndarray, labels: list[str]) -> list[dict]:
    """False rejects and false accepts for each pass score, as practice would see them.

    A false reject is a correct sign scored as a fail. A false accept is a
    clip that would pass as some *other* sign, had that been the target.
    """
    signs = np.array([label != IDLE_LABEL for label in labels])
    wrong = probs * signs
    wrong[np.arange(len(truth)), truth] = 0.0
    top = probs.argmax(axis=1)
    genuine = np.where(top == truth, probs[np.arange(len(truth)), truth], 0.0)
    impostor = np.where(top == wrong.argmax(axis=1), wrong.max(axis=1), 0.0)
    real = signs[truth]  # idle clips are never a practice target
    return [
        {
            "pass_score": round(float(t), 2),
            "false_reject": float((genuine[real] < t).mean()),
            "false_accept": float((impostor >= t).mean()),
        }
        for t in np.arange(0.30, 0.951, 0.05)
    ]


def export(model: nn.Module, out: Path, labels: list[str], pass_score: float | None, probe: np.ndarray):
    """Write the model as one file and move it into place in a single rename.

    A server that connects mid-export sees either the old model or the new
    one, never a partial file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    # The exporter prints a tick mark that a Windows console (cp1252) cannot
    # encode, which would abort the export after training has finished.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "model.onnx"
        example = torch.zeros(1, CLIP_FRAMES, NORMALISED_SIZE)
        torch.onnx.export(model, (example,), str(raw), input_names=["clip"],
                          output_names=["logits"], dynamo=True, verbose=False)
        proto = onnx.load(str(raw))  # pulls any external weight file back in
    proto.metadata_props.add(key=LABELS_KEY, value=json.dumps(labels))
    if pass_score is not None:
        proto.metadata_props.add(key=PASS_SCORE_KEY, value=str(pass_score))
    pending = out.with_name(out.name + ".tmp")
    onnx.save(proto, str(pending))

    # The exported file must give the server the same answer PyTorch gives here.
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(probe)[None]), dim=1)[0].numpy()
    if not np.allclose(expected, OnnxRecogniser(pending).predict(probe), atol=1e-4):
        pending.unlink()
        raise SystemExit("Exported model disagrees with the trained model.")
    os.replace(pending, out)


def train(clips_dir: Path, out: Path, epochs: int, seed: int, test_signer: str | None, runs: Path):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    clips = load_clips(clips_dir)
    if len({c.label for c in clips} - {IDLE_LABEL}) < 2:
        raise SystemExit(f"Need clips for at least two signs; found {len(clips)} clips.")
    train_clips, test_clips, split_note = split(clips, test_signer, rng)
    # The idle class always exists: straddling windows are labelled with it even
    # when nobody has recorded "no sign" takes.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    test_clips = [c for c in test_clips if c.label in labels]
    print(f"{len(train_clips)} training clips, {len(test_clips)} test clips, "
          f"{len(labels) - 1} signs plus no-sign ({split_note})")

    model = SignNet(len(labels))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, epochs + 1):
        x_all, y_all = training_batch(train_clips, labels, rng)
        model.train()
        total = 0.0
        for batch in torch.from_numpy(rng.permutation(len(train_clips))).split(64):
            optimiser.zero_grad()
            loss = loss_fn(model(x_all[batch]), y_all[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        schedule.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}  loss {total / len(train_clips):.3f}")

    report = {"clips": len(clips), "labels": labels, "split": split_note, "epochs": epochs, "seed": seed}
    confusion, pass_score = None, None
    if test_clips:
        probs = probabilities(model, test_clips)
        truth = np.array([labels.index(c.label) for c in test_clips])
        ranked = np.argsort(-probs, axis=1)
        top1 = float((ranked[:, 0] == truth).mean())
        top5 = float((ranked[:, :5] == truth[:, None]).any(axis=1).mean())
        confusion = np.zeros((len(labels), len(labels)), dtype=int)
        np.add.at(confusion, (truth, ranked[:, 0]), 1)
        curve = threshold_curve(probs, truth, labels)
        report |= {"test_clips": len(test_clips), "top1": top1, "top5": top5, "threshold_curve": curve}
        print(f"test top-1 {top1:.1%}  top-5 {top5:.1%}  on {len(test_clips)} clips ({split_note})")
        if sum(c.label != IDLE_LABEL for c in test_clips) >= MIN_CLIPS_FOR_THRESHOLD:
            best = min(curve, key=lambda row: (row["false_reject"] + row["false_accept"], row["pass_score"]))
            pass_score = best["pass_score"]
            print(f"pass score {pass_score} (false reject {best['false_reject']:.1%}, "
                  f"false accept {best['false_accept']:.1%})")
        else:
            print(f"fewer than {MIN_CLIPS_FOR_THRESHOLD} test clips: keeping the server's default pass score")
    report["pass_score"] = pass_score

    # Accuracy says nothing about a sign that had no held-out clip, so name them.
    tested = {c.label for c in test_clips}
    untested = [label for label in labels if label != IDLE_LABEL and label not in tested]
    report["signs_without_test_clips"] = untested
    if untested:
        print(f"WARNING: {len(untested)} of {len(labels) - 1} signs had no held-out clip, so the "
              f"accuracy above does not cover them: {', '.join(untested)}. Record at least 5 clips per sign.")

    export(model, out, labels, pass_score, to_model_input(clips[0].frames))

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
