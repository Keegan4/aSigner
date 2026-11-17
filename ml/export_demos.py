"""Export one recorded clip per sign for the web app's mannequin demo.

For every label with clips (not the idle class) this picks the most typical
take, tidies it and writes web/public/demos/<LABEL>.json, which the lesson,
practice and quiz pages play back as a drawn figure.

Run with: python ml/export_demos.py
Override a poor pick with: --pick A=u1_0123abcd.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import Clip, load_clips, middle_window, to_model_input  # noqa: E402

from app.recognition.features import (  # noqa: E402
    FACE,
    HAND_POINTS,
    IDLE_LABEL,
    LEFT_HAND,
    POSE,
    POSE_POINTS,
    RIGHT_HAND,
)

ROOT = Path(__file__).resolve().parent.parent
ASPECT = 4 / 3  # the camera records 640x480; x is stretched by this so units are square
MIN_FRAMES = 8  # a shorter take is too choppy to demonstrate anything
MAX_GAP = 3  # frames a hand may vanish for and still be drawn through
MIN_HAND_SHARE = 0.15  # a hand seen in fewer frames than this is a stray detection
DECIMALS = 4


def has_pose_throughout(clip: Clip) -> bool:
    return bool(clip.frames[:, POSE:FACE].any(axis=1).all())


def hand_coverage(clip: Clip) -> float:
    """Hands tracked per frame, averaged over the clip: 0 to 2."""
    hands = clip.frames[:, LEFT_HAND:POSE].reshape(len(clip.frames), 2, -1)
    return float(hands.any(axis=2).sum(axis=1).mean())


def medoid(clips: list[Clip]) -> int:
    """Index of the clip most like the label's other clips, as the model sees them.

    Takes that are too short or lose the body are passed over unless every
    take is like that. So is the half of the takes that lost track of the
    hands most: a demo with the hands missing shows little.
    """
    usable = [i for i, c in enumerate(clips) if len(c.frames) >= MIN_FRAMES and has_pose_throughout(c)]
    usable = usable or list(range(len(clips)))
    coverage = {i: hand_coverage(clips[i]) for i in usable}
    typical = float(np.median(list(coverage.values())))
    candidates = [i for i in usable if coverage[i] >= typical]
    inputs = np.stack([to_model_input(middle_window(c)).ravel() for c in clips])
    distances = np.linalg.norm(inputs[candidates][:, None, :] - inputs[None, :, :], axis=2)
    return candidates[int(distances.mean(axis=1).argmin())]


def fill_gaps(points: np.ndarray, present: np.ndarray, max_gap: int | None) -> np.ndarray:
    """Bridge runs of missing frames that lie between two present ones.

    `points` is (time, n, 3). Gaps longer than `max_gap` (None: any length) and
    gaps at either end are left missing. Returns the new presence mask.
    """
    present = present.copy()
    seen = np.flatnonzero(present)
    for before, after in zip(seen[:-1], seen[1:]):
        gap = after - before - 1
        if gap == 0 or (max_gap is not None and gap > max_gap):
            continue
        for step in range(1, gap + 1):
            share = step / (gap + 1)
            points[before + step] = points[before] * (1 - share) + points[after] * share
            present[before + step] = True
    return present


def smooth(points: np.ndarray, present: np.ndarray) -> np.ndarray:
    """A light 1-2-1 blur over time that never blends in a missing frame."""
    out = points.copy()
    for t in np.flatnonzero(present):
        total, weight = points[t] * 2.0, 2.0
        for neighbour in (t - 1, t + 1):
            if 0 <= neighbour < len(points) and present[neighbour]:
                total, weight = total + points[neighbour], weight + 1.0
        out[t] = total / weight
    return out


def tidy(points: np.ndarray, max_gap: int | None, min_share: float = 0.0) -> list[list[float] | None]:
    """One body part across a clip, (time, n, 3) raw -> a JSON row per frame, None where missing."""
    points = points.astype(np.float64)
    present = points.reshape(len(points), -1).any(axis=1)
    if present.mean() < min_share:
        return [None] * len(points)
    present = fill_gaps(points, present, max_gap)
    points = smooth(points, present)
    points[..., 0] *= ASPECT
    rows = np.round(points.reshape(len(points), -1), DECIMALS)
    return [row.tolist() if seen else None for row, seen in zip(rows, present)]


def export(clip: Clip, source: str) -> dict:
    frames = clip.frames
    count = len(frames)
    pose = tidy(frames[:, POSE:FACE].reshape(count, POSE_POINTS, 3), max_gap=None)
    left = tidy(frames[:, LEFT_HAND:RIGHT_HAND].reshape(count, HAND_POINTS, 3), MAX_GAP, MIN_HAND_SHARE)
    right = tidy(frames[:, RIGHT_HAND:POSE].reshape(count, HAND_POINTS, 3), MAX_GAP, MIN_HAND_SHARE)
    # The figure cannot be drawn without a body: drop frames from either end that have none.
    keep = [i for i, row in enumerate(pose) if row is not None]
    first, last = (keep[0], keep[-1] + 1) if keep else (0, 0)
    return {
        "label": clip.label,
        "fps": round(clip.fps or count / 2.0, 2),
        "source": source,
        "pose": pose[first:last],
        "left": left[first:last],
        "right": right[first:last],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=ROOT / "server" / "data" / "clips")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "demos")
    parser.add_argument("--only", help="comma-separated labels to export; default is all of them")
    parser.add_argument("--pick", action="append", default=[], metavar="LABEL=FILE",
                        help="use this clip file for a label instead of the most typical one")
    args = parser.parse_args()

    picks = dict(item.split("=", 1) for item in args.pick)
    only = set(args.only.split(",")) if args.only else None
    paths = sorted(args.clips.glob("*/*.npz"))  # the order load_clips reads them in
    by_label: dict[str, list[tuple[Path, Clip]]] = {}
    for path, clip in zip(paths, load_clips(args.clips)):
        by_label.setdefault(clip.label, []).append((path, clip))

    args.out.mkdir(parents=True, exist_ok=True)
    for label in sorted(by_label):
        if label == IDLE_LABEL or (only and label not in only):
            continue
        takes = by_label[label]
        if label in picks:
            index = next((i for i, (path, _) in enumerate(takes) if path.name == picks[label]), None)
            if index is None:
                sys.exit(f"{label}: no clip named {picks[label]}")
        else:
            index = medoid([clip for _, clip in takes])
        path, clip = takes[index]
        demo = export(clip, path.name)
        target = args.out / f"{label}.json"
        target.write_text(json.dumps(demo, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{label}: {path.name} ({len(demo['pose'])} frames, {target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
