"""Public isolated-sign data, used only to pretrain the sign model.

The source is the "asl-signs" data of Kaggle's Google Isolated Sign Language
Recognition competition (https://www.kaggle.com/competitions/asl-signs): about
94,000 clips of 250 ASL signs by 21 signers, already reduced to MediaPipe
Holistic landmarks. Downloading it needs a Kaggle account that has accepted the
competition rules; read its licence terms there before using it.

    kaggle competitions download -c asl-signs -p data/public/asl-signs

Unzip it so that train.csv sits in data/public/asl-signs/, which is under
ml/data/ and not committed. Unlike the fingerspelling samples it has movement,
both hands and the body. What it does not have: face blendshapes or head angles
(it has a face mesh the app does not use), so the face part stays empty. The
signs are ASL, not SgSL: labels carry an "asl:" prefix so none of them is ever
taken for a sign taught here.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data import Clip

from app.recognition.features import (  # path set up by data.py
    FACE, FRAME_SIZE, HAND_POINTS, LEFT_HAND, POSE, POSE_POINTS, RIGHT_HAND, resample,
)

SOURCE = "asl-signs (Kaggle, Google Isolated Sign Language Recognition)"
LABEL_PREFIX = "asl:"
SIGNER_PREFIX = "asl-signs-"
# Longer clips are thinned to this many frames. The model sees 32 per clip, so
# this keeps room for time cropping while bounding memory.
MAX_FRAMES = 64

# Holistic's part names, where each goes in a raw frame, and how many points the app keeps.
_PARTS = (("left_hand", LEFT_HAND, HAND_POINTS), ("right_hand", RIGHT_HAND, HAND_POINTS), ("pose", POSE, POSE_POINTS))


def sequence_to_clip(path: Path, max_frames: int = MAX_FRAMES) -> np.ndarray | None:
    """One landmark file as a raw clip, or None if it holds no hand or is not what the dataset describes.

    A part with any missing coordinate in a frame is left out of that frame.
    Frames before the first and after the last visible hand are dropped.
    """
    try:
        table = pq.read_table(path)
        index_name = "landmark_index" if "landmark_index" in table.column_names else "landmark_idx"
        frame = np.asarray(table.column("frame").to_pylist(), dtype=np.int64)
        kind = np.asarray(table.column("type").to_pylist(), dtype=str)
        index = np.asarray(table.column(index_name).to_pylist(), dtype=np.int64)
        xyz = np.stack([np.asarray(table.column(c).to_pylist(), dtype=np.float32) for c in "xyz"], axis=1)
    except (OSError, KeyError, ValueError, TypeError):  # pyarrow's errors derive from these
        return None
    if not len(frame):
        return None

    _, time = np.unique(frame, return_inverse=True)
    clip = np.zeros((int(time.max()) + 1, FRAME_SIZE), dtype=np.float32)
    for name, start, count in _PARTS:
        rows = (kind == name) & (index >= 0) & (index < count)
        part = np.full((len(clip), count, 3), np.nan, dtype=np.float32)
        part[time[rows], index[rows]] = xyz[rows]
        part[~np.isfinite(part).all(axis=(1, 2))] = 0.0
        clip[:, start:start + count * 3] = part.reshape(len(clip), -1)

    seen = np.flatnonzero(clip[:, LEFT_HAND:POSE].any(axis=1))
    if not len(seen):
        return None
    clip = clip[seen[0]:seen[-1] + 1]
    # One frame is not a movement: augment() needs at least two to crop and resample.
    if len(clip) < 2:
        return None
    return resample(clip, max_frames) if len(clip) > max_frames else clip


def load_sequences(folder: Path, per_sign: int | None = None, max_frames: int = MAX_FRAMES) -> list[Clip]:
    """Every usable clip listed in `folder`/train.csv, or `per_sign` of each sign.

    With a limit the list is shuffled first (the same way every time), so the
    clips kept for a sign come from across the signers, not from whoever is listed first.
    """
    with open(folder / "train.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if per_sign is not None:
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    clips, taken = [], Counter()
    for row in rows:
        sign = row["sign"]
        if per_sign is not None and taken[sign] >= per_sign:
            continue
        frames = sequence_to_clip(folder / row["path"], max_frames)
        if frames is not None:
            taken[sign] += 1
            clips.append(Clip(frames, LABEL_PREFIX + sign, SIGNER_PREFIX + row["participant_id"]))
    return clips


assert FACE - POSE == POSE_POINTS * 3
