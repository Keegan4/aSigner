"""Raw x in units of image height: converting old clips and why it matters."""
import numpy as np

from data import LEGACY_ASPECT, load_clips, to_height_units

from app.recognition.features import (
    FACE, FRAME_SIZE, HAND_POINTS, NORMALISED_PARTS, POSE, RIGHT_HAND, X_UNITS, normalise,
)

SHAPE = HAND_POINTS * 3


def hand_in_pixels(angle: float) -> np.ndarray:
    """A fixed right hand turned `angle` radians in the picture, as (21, 3) pixel coordinates."""
    shape = np.random.default_rng(5).uniform(-40, 40, (HAND_POINTS, 3))
    cos, sin = np.cos(angle), np.sin(angle)
    turned = shape @ np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]).T
    return turned + (640, 360, 0)


def frame_from_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """What MediaPipe reports: x and z as fractions of width, y as a fraction of height."""
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    frame[RIGHT_HAND:POSE] = (points / (width, height, width)).ravel()
    return frame


def test_height_units_keep_the_palm_frame_steady_as_the_hand_turns_on_a_wide_image():
    width, height = 1280, 720
    detail = slice(NORMALISED_PARTS[1].start + SHAPE, NORMALISED_PARTS[1].stop)

    def palm_frame(angle, convert):
        frame = frame_from_pixels(hand_in_pixels(angle), width, height)[None]
        if convert:
            frame = to_height_units(frame, width / height)
        return normalise(frame[0])[detail]

    np.testing.assert_allclose(palm_frame(0.0, True), palm_frame(0.8, True), atol=1e-4)
    assert not np.allclose(palm_frame(0.0, False), palm_frame(0.8, False), atol=1e-2)


def test_to_height_units_scales_x_and_z_and_keeps_missing_points_zero():
    frames = np.zeros((2, FRAME_SIZE), dtype=np.float32)
    frames[:, RIGHT_HAND:POSE] = 0.5
    frames[:, FACE:] = 0.25  # blendshapes and head angles are not coordinates
    out = to_height_units(frames, 2.0)
    points = out[:, RIGHT_HAND:POSE].reshape(2, -1, 3)
    assert (points[..., 0] == 1.0).all() and (points[..., 1] == 0.5).all() and (points[..., 2] == 1.0).all()
    assert not out[:, :RIGHT_HAND].any()  # the missing left hand
    assert (out[:, FACE:] == 0.25).all()
    assert (frames[:, RIGHT_HAND:POSE] == 0.5).all()  # the input is untouched


def test_clips_without_x_units_are_converted_on_load(tmp_path):
    frames = np.zeros((4, FRAME_SIZE), dtype=np.float32)
    frames[:, RIGHT_HAND:POSE] = 0.5
    (tmp_path / "A").mkdir()
    np.savez_compressed(tmp_path / "A" / "u1_old.npz", frames=frames, label="A", signer="u1")
    np.savez_compressed(tmp_path / "A" / "u1_new.npz", frames=frames, label="A", signer="u1", x_units=X_UNITS)
    new, old = load_clips(tmp_path)  # sorted by name
    np.testing.assert_array_equal(new.frames, frames)
    np.testing.assert_allclose(old.frames[:, RIGHT_HAND], 0.5 * LEGACY_ASPECT)
    np.testing.assert_allclose(old.frames[:, RIGHT_HAND + 1], 0.5)
