"""Public sign data, used only to pretrain the sign model.

ASLNow! fingerspelling
----------------------
The first source is the ASLNow! dataset (https://huggingface.co/datasets/sid220/asl-now-fingerspelling,
MIT licence): one JSON file per sample, in a folder named after its letter,
holding the 21 hand landmarks MediaPipe's web hand landmarker gave for a single
frame. That is the same tracker and the same coordinates the app records.

    git clone https://huggingface.co/datasets/sid220/asl-now-fingerspelling data/public/asl-now-fingerspelling

Keep it under ml/data/, which is not committed. What it does not have: a body,
a face, a note of which hand it is, or any movement. A sample becomes a short
clip of one still hand, so it can teach hand shapes and nothing else, and the
letters that are traced in the air (J, Z) are present as a handshape only.

CASL-W60 (optional)
-------------------
The second source is CASL-W60 Landmarks
(https://huggingface.co/datasets/luciayen/CASL-W60-Landmarks, MIT licence):
5,889 videos of 60 Central African Sign Language signs, as MediaPipe Holistic
pose (33 points) and both hands (21 points each), 64 frames per clip, in two
parquet files with the dataset's own train/test split.

    mkdir -p data/public/casl-w60/data && cd data/public/casl-w60/data
    curl -LO https://huggingface.co/datasets/luciayen/CASL-W60-Landmarks/resolve/main/data/train-00000-of-00001.parquet
    curl -LO https://huggingface.co/datasets/luciayen/CASL-W60-Landmarks/resolve/main/data/test-00000-of-00001.parquet

Unlike ASLNow! it has movement and a body. Its coordinates are already shifted
and scaled, but hands and pose share one frame (each hand's wrist sits on the
pose wrist of the same side) and the normaliser only uses positions relative to
the shoulders and wrist, so that does not matter. It has no face: the face/head
part stays zero, which the model treats as "face not tracked". The labels are
sign IDs (G001...), kept apart from SgSL glosses with a "casl:" prefix.
"""
import json
import math
from pathlib import Path

import numpy as np

from data import Clip

from app.recognition.features import (  # path set up by data.py
    FACE,
    FRAME_SIZE,
    HAND_POINTS,
    LEFT_HAND,
    POSE,
    POSE_POINTS,
    RIGHT_HAND,
)

SOURCE = "asl-now-fingerspelling (MIT)"
CASL_SOURCE = "CASL-W60 landmarks (MIT)"
CASL_PREFIX = "casl:"
_CASL_POSE, _CASL_HAND = 33 * 3, HAND_POINTS * 3
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


def casl_to_clip(frames: np.ndarray) -> np.ndarray:
    """CASL-W60's (time, 225) frames as raw app frames: pose points 0-24, both hands, no face."""
    clip = np.zeros((len(frames), FRAME_SIZE), dtype=np.float32)
    pose = frames[:, :_CASL_POSE].reshape(len(frames), 33, 3)
    clip[:, POSE:FACE] = pose[:, :POSE_POINTS].reshape(len(frames), -1)
    clip[:, LEFT_HAND:RIGHT_HAND] = frames[:, _CASL_POSE:_CASL_POSE + _CASL_HAND]
    clip[:, RIGHT_HAND:POSE] = frames[:, _CASL_POSE + _CASL_HAND:]
    return clip


def load_casl(folder: Path) -> tuple[list[Clip], list[Clip]]:
    """CASL-W60's train and test clips, in the dataset's own split. Empty if it is not there."""
    split = {}
    for name in ("train", "test"):
        files = sorted(folder.glob(f"**/{name}-*.parquet"))
        if not files:
            return [], []
        import pyarrow.parquet as pq  # only needed for this dataset

        clips = []
        for path in files:
            table = pq.read_table(path, columns=["landmarks", "label"])
            for landmarks, label in zip(table.column("landmarks").to_pylist(), table.column("label").to_pylist()):
                frames = np.asarray(landmarks, dtype=np.float32)
                if frames.ndim != 2 or frames.shape[1] != _CASL_POSE + 2 * _CASL_HAND or not np.isfinite(frames).all():
                    continue
                clips.append(Clip(casl_to_clip(frames), CASL_PREFIX + str(label), f"casl-{name}"))
        split[name] = clips
    return split["train"], split["test"]
