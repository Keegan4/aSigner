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

The dataset says who signed each clip, so whole signers are held out to measure
with (load_asl_signs). Reading the files is the slow part, one small parquet
file per clip, so it is spread over every core like augmentation is.
"""
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data import Clip, worker_count

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
        frame = table.column("frame").to_numpy().astype(np.int64)
        kind = table.column("type").to_numpy(zero_copy_only=False).astype(str)
        index = table.column(index_name).to_numpy().astype(np.int64)
        # Missing coordinates arrive as NaN whether the file stores them as NaN or as nulls.
        xyz = np.stack([table.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in "xyz"], axis=1)
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
    # One frame is not a movement: a training window needs at least two.
    if len(clip) < 2:
        return None
    return resample(clip, max_frames) if len(clip) > max_frames else clip


def _read(job: tuple[Path, int]) -> np.ndarray | None:
    return sequence_to_clip(*job)


def _read_all(paths: list[Path], max_frames: int, workers: int | None) -> list[np.ndarray | None]:
    """`sequence_to_clip` for every path, in order, on several cores when there are enough files to be worth it."""
    if workers is None:
        workers = worker_count() if len(paths) >= 500 else 1
    jobs = [(path, max_frames) for path in paths]
    if workers <= 1:
        return [_read(job) for job in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_read, jobs, chunksize=max(1, len(jobs) // (workers * 16))))


def load_sequences(folder: Path, per_sign: int | None = None, max_frames: int = MAX_FRAMES,
                   workers: int | None = None) -> list[Clip]:
    """Every usable clip listed in `folder`/train.csv, or `per_sign` of each sign.

    With a limit the list is shuffled first (the same way every time), so the
    clips kept for a sign come from across the signers, not from whoever is listed first.
    """
    with open(folder / "train.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    spare = defaultdict(list)  # rows over the limit, tried only if a chosen file turns out unusable
    if per_sign is not None:
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
        chosen, seen = [], Counter()
        for row in rows:
            seen[row["sign"]] += 1
            (chosen if seen[row["sign"]] <= per_sign else spare[row["sign"]]).append(row)
        rows = chosen

    def clip(row: dict, frames: np.ndarray) -> Clip:
        return Clip(frames, LABEL_PREFIX + row["sign"], SIGNER_PREFIX + row["participant_id"])

    read = _read_all([folder / row["path"] for row in rows], max_frames, workers)
    clips = [clip(row, frames) for row, frames in zip(rows, read) if frames is not None]
    if per_sign is not None:
        taken = Counter(c.label for c in clips)
        for sign, extra in spare.items():
            for row in extra:
                if taken[LABEL_PREFIX + sign] >= per_sign:
                    break
                frames = sequence_to_clip(folder / row["path"], max_frames)
                if frames is not None:
                    taken[LABEL_PREFIX + sign] += 1
                    clips.append(clip(row, frames))
    return clips


def load_asl_signs(folder: Path, per_sign: int | None = None, held_out: float = 0.15,
                   workers: int | None = None) -> tuple[list[Clip], list[Clip]]:
    """Training and held-out clips, with whole signers held out (a fixed 15% of them, at least one).

    Nothing is returned when there is no train.csv; with a single signer nothing is held out.
    """
    if not (folder / "train.csv").is_file():
        return [], []
    clips = load_sequences(folder, per_sign, workers=workers)
    signers = sorted({c.signer for c in clips})
    if len(signers) < 2:
        return clips, []
    count = max(1, round(len(signers) * held_out))
    out = {signers[i] for i in np.random.default_rng(0).permutation(len(signers))[:count]}
    return [c for c in clips if c.signer not in out], [c for c in clips if c.signer in out]


assert FACE - POSE == POSE_POINTS * 3
