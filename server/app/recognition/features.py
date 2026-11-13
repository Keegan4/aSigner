"""Decode and normalise landmark frames.

This is the only normaliser: the live server and the training code in ml/
both import it, so model inputs cannot drift between training and serving.
The raw layout must match web/src/lib/tracking/features.ts. Raw x and z are
in units of image height (MediaPipe's x times width / height), so on a
non-square image x and y still measure the same distance and turning a hand in
the picture does not change its shape. Clips recorded before that change are
in image-width units; ml/data.py converts them when it loads them.

Each hand's shape part holds 138 values: 63 wrist-relative coordinates, then
75 "detail" values from `hand_detail` (the hand in its own palm frame, and
thumb-tip distances), which make closed fists such as S and M separable.
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
# Stored with each clip (and sent by the browser) when x and z are in units of image height.
X_UNITS = "height"

# Thumb tip to the middle joint, end joint and tip of each finger.
THUMB_TIP = 4
FINGER_JOINTS = (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)
HAND_DETAIL = HAND_POINTS * 3 + len(FINGER_JOINTS)  # 75

# Normalised coordinates: hand shapes (with detail), hand placements, pose,
# face/head. The appended flags follow the same six-part order.
PART_WIDTHS = (63 + HAND_DETAIL, 63 + HAND_DETAIL, 63, 63, 75, 55)
_OFFSETS = np.cumsum((0, *PART_WIDTHS))
NORMALISED_PARTS = tuple(slice(int(a), int(b)) for a, b in zip(_OFFSETS[:-1], _OFFSETS[1:]))
COORDINATE_SIZE = int(_OFFSETS[-1])  # 532
PRESENCE = slice(COORDINATE_SIZE, COORDINATE_SIZE + len(PART_WIDTHS))
NORMALISED_SIZE = PRESENCE.stop  # 538, including six inferred validity flags
# Shape and placement for each hand, then pose and face/head.
REGION_PARTS = ((0, 2), (1, 3), (4,), (5,))

# The label for "not signing": hands at rest or moving between signs. It is a
# model class, not a Sign row, so it never appears in lessons.
IDLE_LABEL = "_idle"

# Every clip is resampled to this many frames before it reaches the model.
CLIP_FRAMES = 32

_L_SHOULDER, _R_SHOULDER = 11, 12
_WRIST, _INDEX_KNUCKLE, _MIDDLE_KNUCKLE, _PINKY_KNUCKLE = 0, 5, 9, 17
_EPS = 1e-6


def decode(payload: bytes) -> np.ndarray:
    """Turn one WebSocket message into a raw frame, rejecting malformed input."""
    if len(payload) != FRAME_SIZE * 4:
        raise ValueError(f"expected {FRAME_SIZE * 4} bytes, got {len(payload)}")
    frame = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(frame).all():
        raise ValueError("frame contains non-finite values")
    return frame


def hand_detail(shape: np.ndarray, left: bool) -> np.ndarray:
    """75 values describing a hand independently of how it is turned.

    `shape` is the hand's wrist-relative, palm-scaled (21, 3) points. The left
    hand is mirrored first, so a left hand gives the same values as the mirror
    image of a right hand. Returns zeros when the palm frame is undefined.

    The palm frame has u from the wrist to the middle knuckle, v from the pinky
    knuckle to the index knuckle (made perpendicular to u) and n = u x v, the
    palm normal. The first 63 values are all 21 points in (u, v, n): whether the
    thumb lies in front of or behind the fingers is always the n axis. The last
    12 are distances from the thumb tip to finger joints.

    This is only rotation invariant if x, y and z share units, which is why
    raw frames carry x and z in units of image height.
    """
    if left:
        shape = shape * np.array([-1.0, 1.0, 1.0], dtype=shape.dtype)
    u = shape[_MIDDLE_KNUCKLE]  # the wrist is the origin
    u_length = float(np.linalg.norm(u))
    if u_length < _EPS:
        return np.zeros(HAND_DETAIL, dtype=np.float32)
    u = u / u_length
    across = shape[_INDEX_KNUCKLE] - shape[_PINKY_KNUCKLE]
    v = across - (across @ u) * u
    v_length = float(np.linalg.norm(v))
    if v_length < _EPS:
        return np.zeros(HAND_DETAIL, dtype=np.float32)
    v = v / v_length
    n = np.cross(u, v)
    in_palm = shape @ np.stack([u, v, n], axis=1)
    distances = np.linalg.norm(shape[list(FINGER_JOINTS)] - shape[THUMB_TIP], axis=1)
    return np.concatenate([in_palm.ravel(), distances]).astype(np.float32)


def normalise(frame: np.ndarray) -> np.ndarray:
    """Make a raw frame independent of where and how large the signer appears.

    Body and hand placement are measured from the midpoint of the shoulders in
    units of shoulder width. Hand shape is measured from the wrist in units of
    palm length. Missing coordinates stay zero. Six appended flags distinguish
    usable parts from missing/degenerate observations before normalisation.
    Each hand shape is followed by its `hand_detail`, which shares its flag.
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

    left_shape, right_shape = hand_shape(left), hand_shape(right)
    return np.concatenate(
        [
            left_shape.ravel(),
            hand_detail(left_shape, left=True),
            right_shape.ravel(),
            hand_detail(right_shape, left=False),
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
