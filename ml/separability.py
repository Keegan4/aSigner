"""How well two signs separate on hand shape alone, measured on recorded clips.

    python separability.py --clips ../server/data/clips S M

Each clip becomes one vector: the hand-shape part of its signing hand (the
hand tracked in more frames), averaged over the frames where it was tracked.
For each feature set it reports leave-one-out nearest-centroid accuracy on
standardised values, where 50% means the two signs cannot be told apart and
100% means every clip lands nearer its own sign. Every feature counts equally
there, so a few informative values among many noisy ones pull it down; the
model can weight them, so treat it as a floor. Then it lists the features that
separate the signs most (difference of means over pooled spread): for S and M,
look for "point 4 palm-n" (thumb tip depth against the palm) near the top.

Any pair works: S/M, M/N, A/S, E/S and T/N are all closed fists. This says
nothing about the model, only whether the information is in the features.
"""
import argparse
from pathlib import Path

import numpy as np

from data import Clip, load_clips

from app.recognition.features import (  # path set up by data.py
    COORDINATE_SIZE, HAND_DETAIL, HAND_POINTS, NORMALISED_PARTS, FINGER_JOINTS, normalise,
)

SHAPE = HAND_POINTS * 3  # the original wrist-relative coordinates, before the detail values
FEATURE_SETS = {
    "original hand shape (63)": slice(0, SHAPE),
    "palm frame and thumb distances (75)": slice(SHAPE, SHAPE + HAND_DETAIL),
    "both (138)": slice(0, SHAPE + HAND_DETAIL),
}


def feature_name(index: int) -> str:
    """A readable name for one value of a hand-shape part."""
    if index < SHAPE:
        point, axis = divmod(index, 3)
        return f"point {point} {'xyz'[axis]}"
    index -= SHAPE
    if index < SHAPE:
        point, axis = divmod(index, 3)
        return f"point {point} palm-{'uvn'[axis]}"
    return f"thumb tip to point {FINGER_JOINTS[index - SHAPE]}"


def hand_vector(frames: np.ndarray) -> np.ndarray | None:
    """Mean hand-shape part of the clip's signing hand, or None if no hand was usable."""
    normalised = np.stack([normalise(frame) for frame in frames])
    best = None
    for hand in (0, 1):
        present = normalised[:, COORDINATE_SIZE + hand] > 0.5
        if present.any() and (best is None or present.sum() > best[1].sum()):
            best = (hand, present)
    if best is None:
        return None
    hand, present = best
    return normalised[present][:, NORMALISED_PARTS[hand]].mean(axis=0)


def vectors(clips: list[Clip], label: str) -> np.ndarray:
    found = [v for v in (hand_vector(c.frames) for c in clips if c.label == label) if v is not None]
    return np.array(found, dtype=np.float64).reshape(len(found), -1)


def leave_one_out_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    """Share of clips nearer their own sign's centroid, each held out in turn.

    Standardisation uses only the clips that are not held out.
    """
    data = np.concatenate([a, b])
    labels = np.array([0] * len(a) + [1] * len(b))
    correct = 0
    for held in range(len(data)):
        keep = np.arange(len(data)) != held
        rest, rest_labels = data[keep], labels[keep]
        mean, spread = rest.mean(axis=0), rest.std(axis=0) + 1e-6
        scaled = (rest - mean) / spread
        probe = (data[held] - mean) / spread
        centroids = [scaled[rest_labels == k].mean(axis=0) for k in (0, 1)]
        guess = int(np.argmin([np.linalg.norm(probe - c) for c in centroids]))
        correct += guess == labels[held]
    return correct / len(data)


def most_separating(a: np.ndarray, b: np.ndarray, count: int = 5) -> list[tuple[str, float]]:
    """Features with the largest |mean difference| / pooled standard deviation."""
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2) + 1e-6
    effect = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled
    order = np.argsort(effect)[::-1][:count]
    return [(feature_name(int(i)), float(effect[i])) for i in order]


def report(a: np.ndarray, b: np.ndarray, first: str, second: str) -> str:
    lines = [f"{first}: {len(a)} clips, {second}: {len(b)} clips",
             "Leave-one-out nearest-centroid accuracy (50% = no better than guessing):"]
    for name, part in FEATURE_SETS.items():
        lines.append(f"  {name:<38} {leave_one_out_accuracy(a[:, part], b[:, part]):6.1%}")
    lines.append("Most separating features (difference of means / pooled spread):")
    for name, effect in most_separating(a, b):
        lines.append(f"  {name:<38} {effect:6.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--clips", type=Path, default=Path(__file__).parent.parent / "server" / "data" / "clips")
    args = parser.parse_args()
    clips = load_clips(args.clips)
    a, b = vectors(clips, args.first), vectors(clips, args.second)
    if len(a) < 2 or len(b) < 2:
        raise SystemExit(f"need at least two clips with a tracked hand for each sign; found "
                         f"{len(a)} for {args.first} and {len(b)} for {args.second} under {args.clips}")
    print(report(a, b, args.first, args.second))


if __name__ == "__main__":
    main()
