"""Synthetic S-like and M-like fists: does the palm frame separate them where raw shape does not?"""
import numpy as np

from data import Clip
from separability import FEATURE_SETS, leave_one_out_accuracy, most_separating, vectors

from app.recognition.features import FRAME_SIZE, HAND_POINTS, POSE, RIGHT_HAND, THUMB_TIP


def fist(thumb_depth: float) -> np.ndarray:
    """A right fist in its palm frame (u, v, n). Only the thumb tip's depth differs between letters."""
    points = np.zeros((HAND_POINTS, 3))
    for knuckle, (u, v) in {5: (0.95, 0.3), 9: (1.0, 0.0), 13: (0.95, -0.25), 17: (0.85, -0.5)}.items():
        points[knuckle] = (u, v, 0)
        points[knuckle + 1 : knuckle + 4] = [(u + 0.1, v, 0.3), (u - 0.15, v, 0.4), (u - 0.3, v, 0.25)]
    points[1:4] = [(0.2, 0.35, 0.05), (0.4, 0.45, 0.1), (0.55, 0.3, 0.2)]
    points[THUMB_TIP] = (0.6, 0.0, thumb_depth)
    return points


def rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def clip(rng: np.random.Generator, label: str, frames: int = 8) -> Clip:
    """The fist held at a random 3D turn, in image coordinates, with tracking noise."""
    shape = fist(0.6 if label == "S" else 0.1) + rng.normal(0, 0.04, (HAND_POINTS, 3))
    turned = shape @ rotation(rng).T * 0.12 + (0.5, 0.55, 0.0)
    raw = np.zeros((frames, FRAME_SIZE), dtype=np.float32)
    raw[:, RIGHT_HAND:POSE] = turned.ravel() + rng.normal(0, 0.002, (frames, HAND_POINTS * 3))
    return Clip(raw, label, "u1")


def test_palm_frame_separates_fists_that_differ_only_in_thumb_depth():
    rng = np.random.default_rng(0)
    clips = [clip(rng, label) for label in "SM" for _ in range(30)]
    s, m = vectors(clips, "S"), vectors(clips, "M")
    original, detail, both = (leave_one_out_accuracy(s[:, part], m[:, part]) for part in FEATURE_SETS.values())
    assert original < 0.65  # turned at random, raw coordinates barely tell them apart
    assert detail > 0.85 and both > original + 0.2
    (name, effect), *_ = most_separating(s, m)
    assert name == "point 4 palm-n" and effect > 5  # the thumb tip's depth against the palm
