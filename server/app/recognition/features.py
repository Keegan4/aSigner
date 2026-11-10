"""Decode and normalise landmark frames.

This is the only normaliser: the live server and the training code in ml/
both import it, so model inputs cannot drift between training and serving.
The raw layout must match web/src/lib/tracking/features.ts.
"""
import numpy as np

HAND_POINTS = 21
POSE_POINTS = 25
BLENDSHAPES = 52

LEFT_HAND = 0
RIGHT_HAND = LEFT_HAND + HAND_POINTS * 3
POSE = RIGHT_HAND + HAND_POINTS * 3
FACE = POSE + POSE_POINTS * 3
HEAD = FACE + BLENDSHAPES
FRAME_SIZE = HEAD + 3  # 256 float32 values, 1024 bytes

# Per hand: shape (wrist-relative) and placement (body-relative).
NORMALISED_SIZE = 4 * HAND_POINTS * 3 + POSE_POINTS * 3 + BLENDSHAPES + 3  # 382

# The label for "not signing": hands at rest or moving between signs. It is a
# model class, not a Sign row, so it never appears in lessons.
IDLE_LABEL = "_idle"

# Every clip is resampled to this many frames before it reaches the model.
CLIP_FRAMES = 32

_L_SHOULDER, _R_SHOULDER = 11, 12
_WRIST, _MIDDLE_KNUCKLE = 0, 9
_EPS = 1e-6


def decode(payload: bytes) -> np.ndarray:
    """Turn one WebSocket message into a raw frame, rejecting malformed input."""
    if len(payload) != FRAME_SIZE * 4:
        raise ValueError(f"expected {FRAME_SIZE * 4} bytes, got {len(payload)}")
    frame = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(frame).all():
        raise ValueError("frame contains non-finite values")
    return frame


def normalise(frame: np.ndarray) -> np.ndarray:
    """Make a raw frame independent of where and how large the signer appears.

    Body and hand placement are measured from the midpoint of the shoulders in
    units of shoulder width. Hand shape is measured from the wrist in units of
    palm length. Parts that were not detected (all zeros) stay all zeros.
    """
    left = frame[LEFT_HAND:RIGHT_HAND].reshape(HAND_POINTS, 3)
    right = frame[RIGHT_HAND:POSE].reshape(HAND_POINTS, 3)
    pose = frame[POSE:FACE].reshape(POSE_POINTS, 3)

    has_pose = bool(pose.any())
    centre = (pose[_L_SHOULDER] + pose[_R_SHOULDER]) / 2
    width = float(np.linalg.norm(pose[_L_SHOULDER, :2] - pose[_R_SHOULDER, :2]))
    has_body = has_pose and width > _EPS

    def to_body(points: np.ndarray) -> np.ndarray:
        if not has_body or not points.any():
            return np.zeros_like(points)
        return (points - centre) / width

    def hand_shape(points: np.ndarray) -> np.ndarray:
        palm = float(np.linalg.norm(points[_MIDDLE_KNUCKLE] - points[_WRIST]))
        if not points.any() or palm < _EPS:
            return np.zeros_like(points)
        return (points - points[_WRIST]) / palm

    return np.concatenate(
        [
            hand_shape(left).ravel(),
            hand_shape(right).ravel(),
            to_body(left).ravel(),
            to_body(right).ravel(),
            to_body(pose).ravel(),
            frame[FACE:],
        ]
    ).astype(np.float32)


def resample(frames: np.ndarray, count: int = CLIP_FRAMES) -> np.ndarray:
    """Pick `count` frames evenly across a clip of any length, oldest first.

    Frames are selected, not blended: averaging a frame where a hand was
    missing (zeros) with one where it was present would invent a hand halfway
    to the origin.
    """
    index = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return np.ascontiguousarray(frames[index], dtype=np.float32)
