import numpy as np
import pytest

from app.recognition.features import (
    FACE,
    COORDINATE_SIZE,
    FINGER_JOINTS,
    FRAME_SIZE,
    HAND_DETAIL,
    HAND_POINTS,
    LEFT_HAND,
    NORMALISED_PARTS,
    NORMALISED_SIZE,
    POSE,
    PRESENCE,
    RIGHT_HAND,
    THUMB_TIP,
    decode,
    hand_detail,
    normalise,
)

SHAPE = HAND_POINTS * 3  # the wrist-relative values that start each hand-shape part


def make_frame(seed: int = 0, right_hand: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.uniform(0.2, 0.8, FRAME_SIZE).astype(np.float32)
    if not right_hand:
        frame[RIGHT_HAND:POSE] = 0
    return frame


def moved(frame: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """The same pose, as if the signer stood elsewhere at a different distance."""
    out = frame.copy()
    out[:FACE] = frame[:FACE] * scale + shift
    return out


def test_layout_matches_the_browser():
    # web/src/lib/tracking/features.ts sends 256 floats per frame.
    assert (LEFT_HAND, RIGHT_HAND, POSE, FACE, FRAME_SIZE) == (0, 63, 126, 201, 256)


def test_decode_round_trip_and_validation():
    frame = make_frame()
    assert np.array_equal(decode(frame.tobytes()), frame)
    with pytest.raises(ValueError):
        decode(b"\x00" * 10)
    bad = frame.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        decode(bad.tobytes())


def test_normalise_ignores_position_and_distance():
    frame = make_frame()
    a = normalise(frame)
    b = normalise(moved(frame, scale=0.5, shift=0.13))
    assert a.shape == (NORMALISED_SIZE,)
    assert np.allclose(a, b, atol=1e-4)


def test_missing_parts_stay_zero():
    out = normalise(make_frame(right_hand=False))
    assert not out[NORMALISED_PARTS[1]].any()  # right hand shape and detail
    assert not out[NORMALISED_PARTS[3]].any()  # right hand placement
    assert out[NORMALISED_PARTS[0]].any() and out[NORMALISED_PARTS[2]].any()  # left hand still present

    empty = normalise(np.zeros(FRAME_SIZE, dtype=np.float32))
    assert not empty.any()
    assert np.isfinite(empty).all()


def test_presence_tracks_normalisation_validity():
    frame = make_frame(right_hand=False)
    assert normalise(frame).shape == (NORMALISED_SIZE,) == (538,)
    np.testing.assert_array_equal(normalise(frame)[PRESENCE], [1, 0, 1, 0, 1, 1])
    # A detected but collapsed hand has placement, but no usable palm scale.
    frame[:63] = 0.5
    np.testing.assert_array_equal(normalise(frame)[PRESENCE], [0, 0, 1, 0, 1, 1])
    frame[POSE:FACE] = 0
    np.testing.assert_array_equal(normalise(frame)[PRESENCE], [0, 0, 0, 0, 0, 1])
    assert not normalise(frame)[:COORDINATE_SIZE - 55].any()


def test_neutral_face_is_ambiguous_but_valid_zero_hand_coordinates_are_not_missing():
    frame = make_frame()
    frame[FACE:] = 0
    out = normalise(frame)
    assert out[COORDINATE_SIZE + 5] == 0
    assert out[COORDINATE_SIZE] == 1
    assert not out[:3].any()  # wrist-relative wrist is zero despite a valid hand


def rotation(seed: int) -> np.ndarray:
    """A random proper 3D rotation."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def turned(frame: np.ndarray, hand: int, matrix: np.ndarray) -> np.ndarray:
    """The frame with one hand rotated in 3D about its wrist."""
    out = frame.copy()
    points = out[hand : hand + SHAPE].reshape(HAND_POINTS, 3)
    points[:] = (points - points[0]) @ matrix.T + points[0]
    return out


def test_hand_detail_ignores_how_the_hand_is_turned():
    frame = make_frame(3)
    for hand, part in ((LEFT_HAND, NORMALISED_PARTS[0]), (RIGHT_HAND, NORMALISED_PARTS[1])):
        before = normalise(frame)[part]
        after = normalise(turned(frame, hand, rotation(hand + 1)))[part]
        assert not np.allclose(before[:SHAPE], after[:SHAPE], atol=1e-2)  # orientation-dependent
        np.testing.assert_allclose(before[SHAPE:], after[SHAPE:], atol=1e-4)
        assert before[SHAPE:].any()


def fist(thumb_depth: float) -> np.ndarray:
    """A right hand in its palm frame (u, v, n), fingers curled towards +n."""
    points = np.zeros((HAND_POINTS, 3))
    knuckles = {5: (0.95, 0.3), 9: (1.0, 0.0), 13: (0.95, -0.25), 17: (0.85, -0.5)}
    for knuckle, (u, v) in knuckles.items():
        points[knuckle] = (u, v, 0)
        points[knuckle + 1 : knuckle + 4] = [(u + 0.1, v, 0.3), (u - 0.15, v, 0.4), (u - 0.3, v, 0.25)]
    points[1:4] = [(0.2, 0.35, 0.05), (0.4, 0.45, 0.1), (0.55, 0.3, 0.2)]
    points[THUMB_TIP] = (0.6, 0.0, thumb_depth)
    return points


def test_thumb_in_front_of_or_behind_the_fingers_is_the_palm_normal():
    matrix = rotation(7)
    in_front, behind = (
        hand_detail(fist(depth) @ matrix.T, left=False)[:SHAPE].reshape(HAND_POINTS, 3)
        for depth in (0.6, 0.1)
    )
    fingers = list(FINGER_JOINTS[::3])  # middle joints
    assert (in_front[THUMB_TIP, 2] - in_front[fingers, 2] > 0).all()
    assert (behind[THUMB_TIP, 2] - behind[fingers, 2] < 0).all()
    np.testing.assert_allclose(in_front[THUMB_TIP, :2], behind[THUMB_TIP, :2], atol=1e-6)


def test_hand_detail_is_zero_without_a_usable_palm():
    assert not hand_detail(np.zeros((HAND_POINTS, 3), np.float32), left=False).any()
    flat = fist(0.3)
    flat[17] = flat[5] = flat[9] * 0.5  # index and pinky knuckles on the wrist-middle line
    assert hand_detail(flat, left=True).shape == (HAND_DETAIL,)
    assert not hand_detail(flat, left=True).any()
    frame = make_frame()
    frame[LEFT_HAND:RIGHT_HAND] = 0.5  # detected but collapsed
    assert not normalise(frame)[NORMALISED_PARTS[0]].any()
