"""export_demos.py on small made-up clips."""
import numpy as np

from data import Clip
from export_demos import ASPECT, export, fill_gaps, medoid, tidy

from app.recognition.features import FRAME_SIZE, HAND_POINTS, LEFT_HAND, POSE, POSE_POINTS, RIGHT_HAND


def clip(hand_x: float, frames: int = 12, label: str = "A") -> Clip:
    """A signer standing still with the right hand at `hand_x`."""
    data = np.zeros((frames, FRAME_SIZE), dtype=np.float32)
    pose = np.full((POSE_POINTS, 3), 0.5, dtype=np.float32)
    pose[11] = (0.6, 0.4, 0.0)
    pose[12] = (0.4, 0.4, 0.0)
    data[:, POSE:POSE + POSE_POINTS * 3] = pose.ravel()
    hand = np.linspace(0, 0.1, HAND_POINTS * 3, dtype=np.float32).reshape(HAND_POINTS, 3)
    hand[:, 0] += hand_x
    data[:, RIGHT_HAND:RIGHT_HAND + HAND_POINTS * 3] = hand.ravel()
    return Clip(data, label, "u1", 12.0)


def test_medoid_is_the_central_clip():
    assert medoid([clip(0.1), clip(0.5), clip(0.3), clip(0.32), clip(0.28)]) == 2


def test_medoid_passes_over_short_takes_and_lost_bodies():
    short = clip(0.3, frames=4)
    bodiless = clip(0.3)
    bodiless.frames[3, POSE:POSE + POSE_POINTS * 3] = 0
    assert medoid([short, bodiless, clip(0.9)]) == 2
    assert medoid([short]) == 0  # nothing better to offer


def test_medoid_prefers_takes_that_kept_the_hands():
    lost = [clip(0.3), clip(0.3)]
    for take in lost:
        take.frames[::2, RIGHT_HAND:POSE] = 0
    assert medoid([*lost, clip(0.1), clip(0.5), clip(0.52)]) in (2, 3, 4)


def test_short_gaps_are_bridged_and_long_ones_left():
    points = np.zeros((9, 1, 3))
    present = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    points[0], points[2], points[7] = 1.0, 3.0, 8.0
    filled = fill_gaps(points, present, max_gap=3)
    assert filled.tolist() == [True, True, True, False, False, False, False, True, False]
    assert points[1, 0, 0] == 2.0


def test_tidy_marks_missing_frames_and_squares_x():
    points = np.ones((6, 2, 3))
    points[5] = 0
    rows = tidy(points, max_gap=3)
    assert rows[5] is None
    assert rows[0] == [round(ASPECT, 4), 1.0, 1.0] * 2


def test_a_stray_hand_is_dropped():
    points = np.zeros((10, 2, 3))
    points[4] = 1
    assert tidy(points, max_gap=3, min_share=0.15) == [None] * 10


def test_export_has_a_row_per_frame_for_every_part():
    source = clip(0.3)
    source.frames[:, LEFT_HAND:RIGHT_HAND] = 0
    demo = export(source, "u1_x.npz")
    assert demo["label"] == "A" and demo["fps"] == 12.0 and demo["source"] == "u1_x.npz"
    assert len(demo["pose"]) == len(demo["left"]) == len(demo["right"]) == 12
    assert len(demo["pose"][0]) == POSE_POINTS * 3 and len(demo["right"][0]) == HAND_POINTS * 3
    assert all(row is None for row in demo["left"])
