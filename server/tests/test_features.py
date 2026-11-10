import numpy as np
import pytest

from app.recognition.features import (
    FACE,
    COORDINATE_SIZE,
    FRAME_SIZE,
    LEFT_HAND,
    NORMALISED_SIZE,
    POSE,
    PRESENCE,
    RIGHT_HAND,
    decode,
    normalise,
)


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
    assert not out[63:126].any()  # right hand shape
    assert not out[189:252].any()  # right hand placement
    assert out[:63].any() and out[126:189].any()  # left hand still present

    empty = normalise(np.zeros(FRAME_SIZE, dtype=np.float32))
    assert not empty.any()
    assert np.isfinite(empty).all()


def test_presence_tracks_normalisation_validity():
    frame = make_frame(right_hand=False)
    assert normalise(frame).shape == (388,)
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
