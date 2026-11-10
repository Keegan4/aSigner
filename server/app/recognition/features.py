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

# Normalised coordinates: hand shapes, hand placements, pose, face/head.
# The appended flags follow the same six-part order.
PART_WIDTHS = (63, 63, 63, 63, 75, 55)
_OFFSETS = np.cumsum((0, *PART_WIDTHS))
NORMALISED_PARTS = tuple(slice(int(a), int(b)) for a, b in zip(_OFFSETS[:-1], _OFFSETS[1:]))
COORDINATE_SIZE = int(_OFFSETS[-1])  # 382
PRESENCE = slice(COORDINATE_SIZE, COORDINATE_SIZE + len(PART_WIDTHS))
NORMALISED_SIZE = PRESENCE.stop  # 388, including six inferred validity flags
# Shape and placement for each hand, then pose and face/head.
REGION_PARTS = ((0, 2), (1, 3), (4,), (5,))

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
    palm length. Missing coordinates stay zero. Six appended flags distinguish
    usable parts from missing/degenerate observations before normalisation.
    An all-zero face/head cannot distinguish a neutral face from missing data.
    """
    left = frame[LEFT_HAND:RIGHT_HAND].reshape(HAND_POINTS, 3)
    right = frame[RIGHT_HAND:POSE].reshape(HAND_POINTS, 3)
    pose = frame[POSE:FACE].reshape(POSE_POINTS, 3)

    has_pose = bool(pose.any())
    centre = (pose[_L_SHOULDER] + pose[_R_SHOULDER]) / 2
    width = float(np.linalg.norm(pose[_L_SHOULDER, :2] - pose[_R_SHOULDER, :2]))
    has_body = has_pose and width > _EPS
    flags = [
        bool(left.any()) and np.linalg.norm(left[_MIDDLE_KNUCKLE] - left[_WRIST]) >= _EPS,
        bool(right.any()) and np.linalg.norm(right[_MIDDLE_KNUCKLE] - right[_WRIST]) >= _EPS,
        has_body and bool(left.any()),
        has_body and bool(right.any()),
        has_body,
        bool(frame[FACE:].any()),
    ]

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
            flags,
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
