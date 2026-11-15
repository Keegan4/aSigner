"""Loading and augmenting recorded clips.

Clips are the .npz files written by the server's /api/clips route: raw frames
of shape (time, 256). Normalisation and resampling come from the server's
features module so training sees exactly what live inference sees.
"""
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.recognition.features import (  # noqa: E402
    CLIP_FRAMES,
    FACE,
    HAND_POINTS,
    HEAD,
    IDLE_LABEL,
    LEFT_HAND,
    POSE,
    POSE_POINTS,
    RIGHT_HAND,
    WINDOW_SECONDS,
    normalise,
    resample,
)

# MediaPipe pose landmarks that trade places when the image is mirrored.
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24)]

WINDOW_SHARE = 0.5  # of training clips shown as a live translator window; the rest whole, as practice sends them
STRADDLE_SHARE = 0.4  # of those windows, how many straddle two clips
SIGN_FRACTION = 0.6  # a straddling window counts as a sign only when that sign fills this much of it
# A training window lasts between half and twice the translator's: it starts
# guessing on half a window, and the range keeps the model from fitting one length.
WINDOW_SCALE = (0.5, 2.0)
# Below this many clips an epoch's augmentation takes well under a second, and
# starting worker processes would cost more than it saves.
MIN_CLIPS_FOR_WORKERS = 2000


# What a clip with no frame rate is taken to last: the length of a /record take.
NOMINAL_CLIP_SECONDS = 2.0


@dataclass
class Clip:
    frames: np.ndarray  # raw, (time, 256)
    label: str
    signer: str
    fps: float | None = None  # as tracked; None when the source does not say


def load_clips(folder: Path) -> list[Clip]:
    clips = []
    for path in sorted(folder.glob("*/*.npz")):
        with np.load(path) as data:
            fps = float(data["fps"]) if "fps" in data else None
            clips.append(Clip(data["frames"].astype(np.float32), str(data["label"]), str(data["signer"]),
                              fps if fps and np.isfinite(fps) else None))
    return clips


def window_frames(clip: Clip, seconds: float = WINDOW_SECONDS) -> int:
    """How many of the clip's frames the live translator would hold in a window of `seconds`.

    The translator keeps every frame no older than the window, so a window
    holds one frame more than the gaps that fit in it. The count follows the
    rate the clip was tracked at: about 6 frames at 10 a second, 16 at 30.
    """
    fps = clip.fps or len(clip.frames) / NOMINAL_CLIP_SECONDS
    return max(2, int(seconds * fps) + 1)


def live_windows(clip: Clip) -> list[np.ndarray]:
    """The clip as the live translator would see it: every window, half a window apart.

    Used to score held-out clips the way translation uses the model, which a
    whole-clip score says nothing about.
    """
    size = window_frames(clip)
    if len(clip.frames) <= size:
        return [clip.frames]
    return [clip.frames[start:start + size] for start in range(0, len(clip.frames) - size + 1, max(1, size // 2))]


def to_model_input(frames: np.ndarray) -> np.ndarray:
    """Raw clip -> (CLIP_FRAMES, NORMALISED_SIZE), the same path the server takes."""
    return resample(np.stack([normalise(frame) for frame in frames]), CLIP_FRAMES)


def _points(frames: np.ndarray) -> np.ndarray:
    """A view of every landmark as (time, 67, 3): left hand, right hand, pose."""
    return frames[:, :FACE].reshape(len(frames), -1, 3)


def mirror(frames: np.ndarray) -> np.ndarray:
    """The clip as a left-right mirror image, so left-handed signing is covered."""
    out = frames.copy()
    points = _points(out)
    present = points.any(axis=2)
    points[..., 0] = np.where(present, 1.0 - points[..., 0], 0.0)
    left = out[:, LEFT_HAND:RIGHT_HAND].copy()
    out[:, LEFT_HAND:RIGHT_HAND] = out[:, RIGHT_HAND:POSE]
    out[:, RIGHT_HAND:POSE] = left
    pose = out[:, POSE:FACE].reshape(len(out), POSE_POINTS, 3)
    for a, b in _POSE_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    out[:, HEAD] *= -1  # yaw
    out[:, HEAD + 2] *= -1  # roll
    return out


def augment(frames: np.ndarray, rng: np.random.Generator, crop: bool = True) -> np.ndarray:
    """A plausible variation of a raw clip. Missing landmarks (zeros) stay missing.

    `crop=False` keeps every part of the clip in place, for a window that was
    already cut to length and labelled by what it holds.
    """
    # Timing: keep a random 70-100% stretch, then drop about one frame in ten.
    out = frames
    if crop:
        keep = max(2, int(len(frames) * rng.uniform(0.7, 1.0)))
        start = rng.integers(0, len(frames) - keep + 1)
        out = frames[start : start + keep]
    if len(out) > 8:
        out = out[rng.random(len(out)) > 0.1]
    out = out.copy()

    if rng.random() < 0.5:
        out = mirror(out)

    # Camera: small roll about the image centre, scale, and sensor noise.
    points = _points(out)
    present = points.any(axis=2, keepdims=True)
    angle = np.deg2rad(rng.uniform(-12, 12))
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = points[..., 0] - 0.5, points[..., 1] - 0.5
    scale = rng.uniform(0.85, 1.15)
    moved = np.stack(
        [(x * cos - y * sin) * scale + 0.5, (x * sin + y * cos) * scale + 0.5, points[..., 2] * scale],
        axis=2,
    )
    moved += rng.normal(0, 0.003, moved.shape)
    points[:] = np.where(present, moved, 0.0)

    # Face features are dropped sometimes so the model cannot lean on one signer's face.
    if rng.random() < 0.3:
        out[:, FACE:] = 0.0
    return out.astype(np.float32)


def window(clip: Clip, other: Clip | None, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """What the live translator shows the model, and what it should be read as.

    The translator slides a short window over continuous signing, so the model
    mostly sees the start, middle or end of a sign rather than all of it, and
    often the end of one sign with the start of the next. Without `other` the
    window is a stretch from anywhere in the clip. With it, the window runs
    across the join between the two clips, in either order, and takes the label
    of whichever fills at least SIGN_FRACTION of it; a window split more evenly
    than that is no sign.
    """
    size = window_frames(clip, WINDOW_SECONDS * rng.uniform(*WINDOW_SCALE))
    frames = clip.frames
    if other is None:
        start = rng.integers(0, max(1, len(frames) - size + 1))
        return frames[start : start + size], clip.label
    own = min(int(rng.integers(1, size)), len(frames))
    rest = min(size - own, len(other.frames))
    if rng.random() < 0.5:  # the other sign ends and this one begins
        joined = np.concatenate([other.frames[len(other.frames) - rest :], frames[:own]])
    else:  # this sign ends and the other begins
        joined = np.concatenate([frames[len(frames) - own :], other.frames[:rest]])
    share = own / len(joined)
    if other.label == clip.label or share >= SIGN_FRACTION:
        return joined, clip.label
    return joined, other.label if 1 - share >= SIGN_FRACTION else IDLE_LABEL


def augmented_sample(clips: list[Clip], index: int, seed: int) -> tuple[np.ndarray, str]:
    """Clip `index` as one training input: augmented, whole or as a live translator window.

    Everything random comes from `seed`, so the result does not depend on which
    process makes it or in what order.
    """
    rng = np.random.default_rng(seed)
    clip = clips[index]
    if rng.random() >= WINDOW_SHARE:
        return to_model_input(augment(clip.frames, rng)), clip.label
    straddles = len(clips) > 1 and rng.random() < STRADDLE_SHARE
    frames, label = window(clip, clips[rng.integers(len(clips))] if straddles else None, rng)
    return to_model_input(augment(frames, rng, crop=False)), label


def worker_count() -> int:
    """Processes to augment with: AUGMENT_WORKERS when set, otherwise every core this process may use."""
    asked = os.environ.get("AUGMENT_WORKERS")
    if asked:
        return max(1, int(asked))
    if hasattr(os, "sched_getaffinity"):  # respects container and taskset limits
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# The pool is kept between epochs: each worker holds its own copy of the clips,
# so only an index and a seed go out per clip and one input array comes back.
_pool: ProcessPoolExecutor | None = None
_pool_for: tuple[list[Clip], int] | None = None
_worker_clips: list[Clip] = []


def _start_worker(clips: list[Clip]) -> None:
    global _worker_clips
    _worker_clips = clips


def _worker_sample(job: tuple[int, int]) -> tuple[np.ndarray, str]:
    return augmented_sample(_worker_clips, *job)


def close_pool() -> None:
    """Stop the augmentation workers, if any are running."""
    global _pool, _pool_for
    if _pool is not None:
        _pool.shutdown(cancel_futures=True)
    _pool = _pool_for = None


def augmented_samples(clips: list[Clip], seeds: np.ndarray, workers: int | None = None) -> list[tuple[np.ndarray, str]]:
    """`augmented_sample` for every clip, in order, spread over `workers` processes.

    By default small sets are done in this process. The answer is the same for
    any number of workers.
    """
    global _pool, _pool_for
    if workers is None:
        workers = worker_count() if len(clips) >= MIN_CLIPS_FOR_WORKERS else 1
    workers = min(workers, len(clips))
    jobs = [(index, int(seed)) for index, seed in enumerate(seeds)]
    if workers <= 1:
        return [augmented_sample(clips, *job) for job in jobs]
    if _pool is None or _pool_for is None or _pool_for[0] is not clips or _pool_for[1] != workers:
        close_pool()
        # On Linux, fork: the workers share the clips already in memory instead of
        # each being sent a copy, which for the public sets is several gigabytes.
        # They only run numpy, so the threads PyTorch has started do not matter.
        context = multiprocessing.get_context("fork" if sys.platform == "linux" else None)
        _pool = ProcessPoolExecutor(workers, mp_context=context, initializer=_start_worker, initargs=(clips,))
        _pool_for = (clips, workers)
    return list(_pool.map(_worker_sample, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


assert HAND_POINTS * 2 + POSE_POINTS == 67
