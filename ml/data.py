"""Loading and augmenting recorded clips.

Clips are the .npz files written by the server's /api/clips route: raw frames
of shape (time, 256). Normalisation and resampling come from the server's
features module so training sees exactly what live inference sees.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.recognition.features import (  # noqa: E402
    CLIP_FRAMES,
    FACE,
    HAND_POINTS,
    HEAD,
    LEFT_HAND,
    POSE,
    POSE_POINTS,
    RIGHT_HAND,
    normalise,
    resample,
)

# MediaPipe pose landmarks that trade places when the image is mirrored.
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24)]


@dataclass
class Clip:
    frames: np.ndarray  # raw, (time, 256)
    label: str
    signer: str


def load_clips(folder: Path) -> list[Clip]:
    clips = []
    for path in sorted(folder.glob("*/*.npz")):
        with np.load(path) as data:
            clips.append(Clip(data["frames"].astype(np.float32), str(data["label"]), str(data["signer"])))
    return clips


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


def augment(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible variation of a raw clip. Missing landmarks (zeros) stay missing."""
    # Timing: keep a random 70-100% stretch, then drop about one frame in ten.
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


def transition(sign: np.ndarray, before: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """What a live two-second window often holds: the end of something else, then this sign.

    Returns the joined frames and the fraction of them that show the sign. The
    translator slides a window over continuous signing, so most windows straddle
    two signs; training on whole clips alone would never show the model that.
    """
    fraction = rng.uniform(0.3, 1.0)
    shown = max(2, int(len(sign) * fraction))
    lead = min(len(before), int(round(shown * (1 - fraction) / fraction)))
    frames = np.concatenate([before[len(before) - lead :], sign[:shown]])
    return frames, shown / len(frames)


assert HAND_POINTS * 2 + POSE_POINTS == 67
