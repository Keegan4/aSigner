"""Train the sign classifier and export it for the server.

    python train.py --clips ../server/data/clips --out ../server/models/signs.onnx
    python train.py --clips ../server/data/clips data/collected   # several folders

data/collected is where pull_samples.py puts takes from the training page.

Writes signs.onnx (labels and pass score inside it), plus report.json and
confusion.csv under runs/<time>/.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from checkpoint import load_into
from data import Clip, augmented_samples, live_windows, load_clips, middle_window, to_model_input

from app.recognition.features import (  # path set up by data.py
    CLIP_FRAMES, COORDINATE_SIZE, HAND_POINTS, IDLE_LABEL, NORMALISED_PARTS,
    NORMALISED_SIZE, PART_WIDTHS, REGION_PARTS,
)
from app.recognition.onnx_recogniser import INFO_KEY, LABELS_KEY, PASS_SCORE_KEY, OnnxRecogniser

MIN_CLIPS_FOR_THRESHOLD = 30
MAX_FALSE_ACCEPT = 0.05
MIN_PASS_SCORE = 0.5
# Each region's share of the embedding, in REGION_PARTS order: left hand,
# right hand, pose, face/head. Hands carry most of a sign, so they get the
# room to keep fine handshape detail (120, 120, 40, 40 at width 320).
REGION_SHARES = (3, 3, 1, 1)
# The bones of MediaPipe's hand: landmark 0 is the wrist, then four landmarks
# per finger from the thumb to the little finger, base to tip.
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def pick_device() -> torch.device:
    """A GPU when PyTorch can see one, otherwise the CPU.

    "cuda" also covers AMD GPUs on a ROCm build of PyTorch; "xpu" is Intel
    Arc and Core Ultra graphics on an XPU build; "mps" is Apple Silicon.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hand_adjacency() -> torch.Tensor:
    """How much each hand landmark takes from each other one in a graph convolution.

    A landmark listens to itself and to the landmarks it shares a bone with.
    Each link is divided by the square roots of both ends' link counts
    (D^-1/2 (A + I) D^-1/2), so the wrist, with the most links, does not drown
    out a fingertip.
    """
    links = torch.eye(HAND_POINTS)
    for a, b in HAND_BONES:
        links[a, b] = links[b, a] = 1.0
    scale = links.sum(dim=1).rsqrt()
    return scale[:, None] * links * scale[None, :]


class HandGraph(nn.Module):
    """Two graph convolutions over the hand skeleton, then one vector for the hand.

    Each layer lets a landmark see one bone further, so after two a fingertip
    knows about the two joints below it. The flat encoder has to learn which
    landmarks are connected; here it is built in.
    """

    def __init__(self, width: int, joint_features: int, hidden: int = 32):
        super().__init__()
        # Fixed, so it is not a weight and stays out of checkpoints.
        self.register_buffer("adjacency", hand_adjacency(), persistent=False)
        self.first = nn.Linear(joint_features, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.out = nn.Linear(HAND_POINTS * hidden, width)

    def mix(self, joints: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """One graph convolution: the same weights for every landmark, then its neighbours' sum."""
        return nn.functional.gelu(self.adjacency @ layer(joints))

    def forward(self, joints: torch.Tensor) -> torch.Tensor:  # (batch, time, landmarks, features)
        joints = self.mix(self.mix(joints, self.first), self.second)
        return self.out(joints.flatten(start_dim=-2))


class RegionEncoder(nn.Module):
    """Encode positions, valid motion and availability for one anatomical region.

    With `graph`, a hand also goes through graph convolutions over its skeleton
    (HandGraph), added to the flat encoder's output.
    """

    def __init__(self, parts: tuple[int, ...], width: int, graph: bool = False):
        super().__init__()
        self.parts = parts
        coordinates = sum(PART_WIDTHS[i] for i in parts)
        inputs = 2 * coordinates + 2 * len(parts)
        # Two layers: the first mixes positions, motion and flags into half the
        # input size; the second brings that to the region's width
        # (e.g. 256 -> 128 -> 120 for a hand).
        self.encoder = nn.Sequential(
            nn.Linear(inputs, inputs // 2), nn.GELU(),
            nn.Linear(inputs // 2, width), nn.GELU(), nn.LayerNorm(width),
        )
        self.missing = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.missing, std=0.02)
        self.graph = None
        if graph:
            if any(PART_WIDTHS[i] != HAND_POINTS * 3 for i in parts):
                raise ValueError("graph convolution is only defined for the hand regions")
            # Per landmark: xyz and its motion, for each of the hand's parts.
            self.graph = HandGraph(width, joint_features=2 * len(parts) * 3)

    def inputs(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions, motions, present, motion_valid = [], [], [], []
        for i in self.parts:
            flag = (clip[..., COORDINATE_SIZE + i:COORDINATE_SIZE + i + 1] > 0.5).to(clip.dtype)
            values = clip[..., NORMALISED_PARTS[i]] * flag
            previous = torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)
            previous_flag = torch.cat([torch.zeros_like(flag[:, :1]), flag[:, :-1]], dim=1)
            valid = flag * previous_flag
            positions.append(values)
            motions.append((values - previous) * valid)
            present.append(flag)
            motion_valid.append(valid)
        flags = torch.cat(present, dim=-1)
        return torch.cat([*positions, *motions, flags, *motion_valid], dim=-1), flags

    def joints(self, features: torch.Tensor) -> torch.Tensor:
        """A hand's positions and motions regrouped by landmark: (batch, time, 21, 12)."""
        groups = 2 * len(self.parts)  # positions then motions, one group per part
        coordinates = features[..., :groups * HAND_POINTS * 3]
        return coordinates.unflatten(-1, (groups, HAND_POINTS, 3)).transpose(-3, -2).flatten(start_dim=-2)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        features, flags = self.inputs(clip)
        encoded = self.encoder(features)
        if self.graph is not None:
            encoded = encoded + self.graph(self.joints(features))
        return torch.where(flags.sum(dim=-1, keepdim=True) > 0, encoded, self.missing)


class SignNet(nn.Module):
    """Attend to all frames, then classify a learned summary token.

    Each normalised frame is one token. Learned positions preserve frame order;
    the prepended classification token gathers information from the whole clip.
    """

    def __init__(self, classes: int, width: int = 320, temporal_conv: bool = True, graph_conv: bool = False):
        super().__init__()
        if width % 20 or width % sum(REGION_SHARES):
            raise ValueError(f"embedding width must be divisible by 20 heads and by {sum(REGION_SHARES)} "
                             "(the regions' shares)")
        # Graph convolution is optional and only for the hands, the regions made
        # of hand landmarks; ablate.py measures what it adds.
        self.regions = nn.ModuleList(
            RegionEncoder(parts, width * share // sum(REGION_SHARES),
                          graph=graph_conv and all(PART_WIDTHS[i] == HAND_POINTS * 3 for i in parts))
            for parts, share in zip(REGION_PARTS, REGION_SHARES)
        )
        # The residual temporal convolution can be left out to measure what it adds
        # (ablate.py); attention can still relate neighbouring frames without it.
        self.temporal_conv = temporal_conv
        if temporal_conv:
            self.temporal_norm = nn.LayerNorm(width)
            self.temporal = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width),
                nn.Conv1d(width, width, kernel_size=1), nn.GELU(), nn.Dropout(0.1),
            )
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.positions = nn.Parameter(torch.empty(1, CLIP_FRAMES + 1, width))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positions, std=0.02)
        # Construct each layer separately so they start with different weights.
        self.encoder = nn.Sequential(*(
            nn.TransformerEncoderLayer(
                d_model=width, nhead=20, dim_feedforward=width * 2,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(2)
        ))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(width // 2, classes),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:  # (batch, time, features)
        frames = torch.cat([region(clip) for region in self.regions], dim=-1)
        if self.temporal_conv:
            local = self.temporal(self.temporal_norm(frames).transpose(1, 2)).transpose(1, 2)
            frames = frames + local
        cls = self.cls_token.expand(clip.shape[0], -1, -1)
        tokens = torch.cat([cls, frames], dim=1) + self.positions
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


def clip_source(signer: str) -> str:
    """"recorded" for clips made in the app (/record, or /train via pull_samples.py),
    otherwise the outside source's id."""
    return "recorded" if re.fullmatch(r"u\d+|t[0-9a-f]{16}", signer) else signer


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


def training_batch(clips: list[Clip], labels: list[str], rng: np.random.Generator, workers: int | None = None):
    """One epoch's inputs: a fresh window of every clip, augmented, some straddling two clips.

    Augmenting is the slow part of an epoch and runs on the CPU, so large sets
    are spread over every core (see data.augmented_samples).
    """
    seeds = rng.integers(np.iinfo(np.int64).max, size=len(clips))
    samples = augmented_samples(clips, seeds, workers)
    index = {label: i for i, label in enumerate(labels)}
    xs = np.stack([x for x, _ in samples])
    return torch.from_numpy(xs), torch.tensor([index[label] for _, label in samples])


def model_inputs(clips: list[Clip]) -> np.ndarray:
    """The middle window of each clip, unaugmented: one input per clip, at the length the model is always shown.

    Worth keeping when the same clips are scored every epoch.
    """
    return np.stack([to_model_input(middle_window(c)) for c in clips])


def probabilities(model: nn.Module, clips: list[Clip] | np.ndarray) -> np.ndarray:
    """Class probabilities for clips, or for inputs already made by `model_inputs`."""
    device = next(model.parameters()).device
    x = torch.from_numpy(clips if isinstance(clips, np.ndarray) else model_inputs(clips)).to(device)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).cpu().numpy()


def window_accuracy(model: nn.Module, clips: list[Clip], labels: list[str]) -> tuple[float, int]:
    """Top-1 over every live translator window of the clips, and how many windows that was."""
    windows = [(window, labels.index(clip.label)) for clip in clips for window in live_windows(clip)]
    probs = probabilities(model, np.stack([to_model_input(window) for window, _ in windows]))
    return float((probs.argmax(axis=1) == np.array([truth for _, truth in windows])).mean()), len(windows)


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


def export(model: nn.Module, out: Path, labels: list[str], pass_score: float | None,
           probe: np.ndarray, info: dict | None = None):
    """Write the model as one file and move it into place in a single rename.

    A server that connects mid-export sees either the old model or the new
    one, never a partial file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # The server runs on the CPU, so the model is exported and checked there.
    model.cpu().eval()
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
    if info is not None:
        proto.metadata_props.add(key=INFO_KEY, value=json.dumps(info))
    pending = out.with_name(out.name + ".tmp")
    onnx.save(proto, str(pending))

    # The exported file must give the server the same answer PyTorch gives here.
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(probe)[None]), dim=1)[0].numpy()
    if not np.allclose(expected, OnnxRecogniser(pending).predict(probe), atol=1e-4):
        pending.unlink()
        raise SystemExit("Exported model disagrees with the trained model.")
    os.replace(pending, out)


def train(clips_dir: Path | list[Path], out: Path, epochs: int, seed: int, test_signer: str | None, runs: Path,
          init: Path | None = None, temporal_conv: bool = True, graph_conv: bool = False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    folders = clips_dir if isinstance(clips_dir, list) else [clips_dir]
    clips = [clip for folder in folders for clip in load_clips(folder)]
    if len({c.label for c in clips} - {IDLE_LABEL}) < 2:
        raise SystemExit(f"Need clips for at least two signs; found {len(clips)} clips.")
    train_clips, test_clips, split_note = split(clips, test_signer, rng)
    if len(train_clips) < 2:
        raise SystemExit("Need at least two training clips for batch normalisation; "
                         "record more clips or choose a different test signer.")
    # The idle class always exists: windows split evenly between two signs are
    # labelled with it even when nobody has recorded "no sign" takes.
    labels = sorted({c.label for c in train_clips} | {IDLE_LABEL})
    test_clips = [c for c in test_clips if c.label in labels]
    print(f"{len(train_clips)} training clips, {len(test_clips)} test clips, "
          f"{len(labels) - 1} signs plus no-sign ({split_note})")

    model = SignNet(len(labels), temporal_conv=temporal_conv, graph_conv=graph_conv)
    if not temporal_conv:
        print("without the residual temporal convolution")
    if graph_conv:
        print("with graph convolution on the hands")
    # Optionally start from weights pretrained on public data (pretrain.py).
    pretrained = load_into(model, labels, init) if init else None
    if pretrained:
        print(f"starting from {init} ({pretrained})")
    device = pick_device()
    model.to(device)
    print(f"training on {device}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, epochs + 1):
        x_all, y_all = training_batch(train_clips, labels, rng)
        x_all, y_all = x_all.to(device), y_all.to(device)
        model.train()
        total = 0.0
        # Balanced batches stay at most 64 and never leave a singleton, which
        # BatchNorm cannot train on (e.g. 65 clips become batches of 33 and 32).
        batches = torch.from_numpy(rng.permutation(len(train_clips))).tensor_split(
            (len(train_clips) + 63) // 64
        )
        for batch in batches:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x_all[batch]), y_all[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        schedule.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}  loss {total / len(train_clips):.3f}")

    report = {"clips": len(clips), "labels": labels, "split": split_note, "epochs": epochs, "seed": seed,
              "temporal_conv": temporal_conv, "graph_conv": graph_conv}
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
        # The figures above are on the middle window of each clip, as practice sends
        # it. The translator's window also lands on the start and end of a sign.
        window_top1, windows = window_accuracy(model, test_clips, labels)
        report |= {"window_top1": window_top1, "windows": windows}
        print(f"as live translator windows: top-1 {window_top1:.1%} on {windows} windows")
        if sum(c.label != IDLE_LABEL for c in test_clips) >= MIN_CLIPS_FOR_THRESHOLD:
            # Passing a wrong sign teaches the wrong sign; failing a right one only
            # costs a retry. So take the most lenient score that keeps false accepts
            # under the cap, never below the floor, rather than balancing the two.
            allowed = [row for row in curve
                       if row["pass_score"] >= MIN_PASS_SCORE and row["false_accept"] <= MAX_FALSE_ACCEPT]
            best = allowed[0] if allowed else curve[-1]
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

    # Travels inside the model so the app can show learners how much it was trained on.
    info = {
        "clips": len(train_clips),
        "signers": len({c.signer for c in train_clips}),
        "top1": report.get("top1"),
        "test_clips": len(test_clips),
        "untested_signs": len(untested),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Where the training clips came from, so the file says so itself: clips
        # recorded in the app carry a "u<number>" signer id, anything converted
        # from outside footage carries the id it was converted with.
        "sources": dict(sorted(Counter(clip_source(c.signer) for c in train_clips).items())),
    }
    # Per sign, so the translate page can show which words are well covered and
    # which the overall accuracy hides: training clips, held-out clips, and the
    # share of those held-out clips read as the right sign (None when untested).
    train_counts = Counter(c.label for c in train_clips)
    test_counts = Counter(c.label for c in test_clips)
    info["signs"] = {
        label: {
            "clips": train_counts[label],
            "test_clips": test_counts[label],
            "top1": (float(confusion[i, i] / test_counts[label])
                     if confusion is not None and test_counts[label] else None),
        }
        for i, label in enumerate(labels)
    }
    if pretrained:
        info["pretrained"] = pretrained
    if not temporal_conv:
        info["temporal_conv"] = False
    if graph_conv:
        info["graph_conv"] = True
    report["info"] = info
    export(model, out, labels, pass_score, to_model_input(middle_window(clips[0])), info)

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
    parser.add_argument("--clips", type=Path, nargs="+", default=[here.parent / "server" / "data" / "clips"],
                        help="one or more clip folders (default: the /record clips)")
    parser.add_argument("--out", type=Path, default=here.parent / "server" / "models" / "signs.onnx")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-signer", help="signer id to hold out (default: the one with fewest clips)")
    parser.add_argument("--runs", type=Path, default=here / "runs")
    parser.add_argument("--init", type=Path, help="checkpoint from pretrain.py to start from")
    parser.add_argument("--no-temporal-conv", dest="temporal_conv", action="store_false",
                        help="leave out the residual temporal convolution")
    parser.add_argument("--graph-conv", action="store_true",
                        help="add graph convolution over each hand's skeleton")
    args = parser.parse_args()
    train(args.clips, args.out, args.epochs, args.seed, args.test_signer, args.runs, args.init,
          args.temporal_conv, args.graph_conv)
