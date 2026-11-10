"""Public fingerspelling data, used only to pretrain the sign model.

The source is the ASLNow! dataset (https://huggingface.co/datasets/sid220/asl-now-fingerspelling,
MIT licence): one JSON file per sample, in a folder named after its letter,
holding the 21 hand landmarks MediaPipe's web hand landmarker gave for a single
frame. That is the same tracker and the same coordinates the app records.

    git clone https://huggingface.co/datasets/sid220/asl-now-fingerspelling data/public/asl-now-fingerspelling

Keep it under ml/data/, which is not committed. What it does not have: a body,
a face, a note of which hand it is, or any movement. A sample becomes a short
clip of one still hand, so it can teach hand shapes and nothing else, and the
letters that are traced in the air (J, Z) are present as a handshape only.
"""
import json
import math
from pathlib import Path

import numpy as np

from data import Clip

from app.recognition.features import FRAME_SIZE, HAND_POINTS, POSE, RIGHT_HAND  # path set up by data.py

SOURCE = "asl-now-fingerspelling (MIT)"
SIGNER = "public"
FRAMES_PER_CLIP = 12


def _landmarks(path: Path) -> np.ndarray | None:
    """The 21 points in a sample file, or None if the file is not what the dataset describes."""
    try:
        points = json.loads(path.read_text(encoding="utf-8"))
        values = [[float(p["x"]), float(p["y"]), float(p["z"])] for p in points]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if len(values) != HAND_POINTS or not all(math.isfinite(v) for point in values for v in point):
        return None
    return np.array(values, dtype=np.float32)


def sample_to_clip(points: np.ndarray, frames: int = FRAMES_PER_CLIP) -> np.ndarray:
    """A raw clip of one still hand. Augmentation mirrors it into the other hand's slot half the time."""
    clip = np.zeros((frames, FRAME_SIZE), dtype=np.float32)
    clip[:, RIGHT_HAND:POSE] = points.ravel()
    return clip


def load_samples(folder: Path) -> list[Clip]:
    """Every usable sample under `folder`, one clip each, labelled with its folder's letter."""
    clips = []
    for path in sorted(folder.glob("*/*.json")):
        label = path.parent.name
        if len(label) != 1 or not "A" <= label <= "Z":
            continue
        points = _landmarks(path)
        if points is not None:
            clips.append(Clip(sample_to_clip(points), label, SIGNER))
    return clips
