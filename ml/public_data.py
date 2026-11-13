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

WLASL landmarks (optional, academic and non-commercial use only)
----------------------------------------------------------------
The third source is a MediaPipe version of WLASL
(https://huggingface.co/datasets/Kibalama/poseformer-sign-language): 11,980
videos of 2,000 American Sign Language words, up to 150 frames each, under
WLASL's Computational Use of Data Agreement (academic, non-commercial use).

    mkdir -p data/public/wlasl-landmarks/data && cd data/public/wlasl-landmarks/data
    for i in 0 1 2 3; do curl -LO https://huggingface.co/datasets/Kibalama/poseformer-sign-language/resolve/main/data/train-0000$i-of-00004.parquet; done

Each frame has 76 points, in an order the dataset does not document. Worked
out from where they sit and which go missing together: 25 face-mesh points,
9 pose points (MediaPipe 0, 11-16, 23, 24), the left hand, the right hand; each
hand's wrist sits on the pose wrist of the same side. The other 16 pose points
are filled from their neighbours so they do not land somewhere odd after
normalisation: eyes, ears and mouth (1-10) at the nose; pinky, index and thumb
(17-22) at the hand's knuckles and thumb when it is tracked, else at the wrist.
The face-mesh points are dropped (the app uses face blendshapes, which this
does not have), so the face counts as not tracked. There is no official split
here and no signer IDs: for every word with at least five clips, a fixed 15%
(at least one) is held out. Labels get a "wlasl:" prefix.
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
WLASL_SOURCE = "WLASL landmarks (C-UDA, academic non-commercial)"
WLASL_PREFIX = "wlasl:"
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


_WLASL_POSE = (0, 11, 12, 13, 14, 15, 16, 23, 24)  # MediaPipe pose index of points 25-33
_WLASL_FACE, _WLASL_LEFT, _WLASL_RIGHT = 25, 34, 55
# Pose pinky, index and thumb for each side, and the hand points nearest them.
_HAND_ENDS = {15: (17, 19, 21), 16: (18, 20, 22)}
_HAND_NEAREST = (17, 5, 4)


def wlasl_to_clip(frames: np.ndarray) -> np.ndarray:
    """WLASL's (time, 76, 3) frames as raw app frames: 25 pose points, both hands, no face."""
    count = len(frames)
    pose = np.zeros((count, POSE_POINTS, 3), dtype=np.float32)
    pose[:, list(_WLASL_POSE)] = frames[:, _WLASL_FACE:_WLASL_LEFT]
    pose[:, 1:11] = pose[:, :1]  # eyes, ears and mouth: at the nose
    hands = {15: frames[:, _WLASL_LEFT:_WLASL_RIGHT], 16: frames[:, _WLASL_RIGHT:]}
    for wrist, ends in _HAND_ENDS.items():
        seen = hands[wrist].any(axis=(1, 2))
        for end, nearest in zip(ends, _HAND_NEAREST):
            pose[:, end] = np.where(seen[:, None], hands[wrist][:, nearest], pose[:, wrist])
    pose[~frames[:, _WLASL_FACE:_WLASL_LEFT].any(axis=(1, 2))] = 0  # no body tracked: keep it missing
    clip = np.zeros((count, FRAME_SIZE), dtype=np.float32)
    clip[:, POSE:FACE] = pose.reshape(count, -1)
    clip[:, LEFT_HAND:RIGHT_HAND] = hands[15].reshape(count, -1)
    clip[:, RIGHT_HAND:POSE] = hands[16].reshape(count, -1)
    return clip


def load_wlasl(folder: Path, held_out: float = 0.15) -> tuple[list[Clip], list[Clip]]:
    """WLASL training and held-out clips. The split is fixed (its own seed), so every run tests on the same clips."""
    files = sorted(folder.glob("**/*.parquet"))
    if not files:
        return [], []
    import pyarrow.parquet as pq  # only needed for this dataset

    by_label: dict[str, list[np.ndarray]] = {}
    for path in files:
        table = pq.read_table(path, columns=["landmarks", "label"])
        for landmarks, label in zip(table.column("landmarks").to_pylist(), table.column("label").to_pylist()):
            frames = np.asarray(landmarks, dtype=np.float32)
            if label in (None, "", "unknown") or frames.ndim != 3 or frames.shape[1:] != (76, 3) \
                    or len(frames) < 2 or not np.isfinite(frames).all():
                continue
            by_label.setdefault(str(label), []).append(wlasl_to_clip(frames))
    rng = np.random.default_rng(12345)
    train, test = [], []
    for label in sorted(by_label):
        group = by_label[label]
        cut = max(1, round(len(group) * held_out)) if len(group) >= 5 else 0
        order = rng.permutation(len(group))
        test += [Clip(group[i], WLASL_PREFIX + label, "wlasl-test") for i in order[:cut]]
        train += [Clip(group[i], WLASL_PREFIX + label, "wlasl") for i in order[cut:]]
    return train, test
