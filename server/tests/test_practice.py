from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import select

from app.features import FRAME_SIZE, HAND_POINTS, NORMALISED_SIZE, POSE, RIGHT_HAND
from app.main import app
from app.models import Attempt, Progress, Sign
from app.practice import MAX_MASTERY, REVIEW_DAYS, record_attempt, score_clip
from app.recogniser import get_recogniser

ORIGIN = {"origin": "http://localhost:3000"}
LABELS = list("ABC")


class FakeRecogniser:
    """Reads the answer from the clip: the right index fingertip's raw x picks the label."""

    labels = LABELS

    def __init__(self):
        self.clips = []

    def predict(self, frames):
        assert frames.ndim == 2 and frames.shape[1] == NORMALISED_SIZE
        self.clips.append(frames)
        # Hand shape is wrist-relative in palm lengths, so this survives normalise.
        marker = frames[:, HAND_POINTS * 3 + 8 * 3].mean()
        out = np.full(len(LABELS), 0.05)
        out[int(round(marker)) % len(LABELS)] = 0.9
        return out


def recorded_frame(label_index: int) -> bytes:
    """A raw frame with shoulders and a right hand whose shape encodes a label."""
    frame = np.zeros(FRAME_SIZE, dtype="<f4")
    pose = frame[POSE : POSE + 75].reshape(25, 3)
    pose[11] = (0.6, 0.5, 0)
    pose[12] = (0.4, 0.5, 0)
    hand = frame[RIGHT_HAND : RIGHT_HAND + 63].reshape(21, 3)
    hand[0] = (0.5, 0.5, 0)  # wrist
    hand[9] = (0.5, 0.4, 0)  # palm length 0.1
    hand[8] = (0.5 + 0.1 * label_index, 0.5, 0)
    return frame.tobytes()


@pytest.fixture
def recogniser(client):
    fake = FakeRecogniser()
    app.dependency_overrides[get_recogniser] = lambda: fake
    return fake


def sign_id(db_factory, gloss: str) -> int:
    with db_factory() as db:
        return db.scalar(select(Sign.id).where(Sign.gloss == gloss))


def attempt(ws, target_id: int, label_index: int, frames: int = 10) -> dict:
    ws.send_json({"type": "start", "sign_id": target_id})
    assert ws.receive_json()["type"] == "ready"
    for _ in range(frames):
        ws.send_bytes(recorded_frame(label_index))
    ws.send_json({"type": "score"})
    return ws.receive_json()


def test_replayed_frames_are_scored_and_stored(signed_in, user, db_factory, recogniser):
    b = sign_id(db_factory, "B")
    with signed_in.websocket_connect("/ws/practice", headers=ORIGIN) as ws:
        result = attempt(ws, b, LABELS.index("B"))
        assert result["type"] == "result"
        assert result["passed"] is True and result["closest"] is None
        assert result["score"] == pytest.approx(0.9)
        assert (result["mastery"], result["streak"]) == (1, 1)
        assert recogniser.clips[0].shape == (10, NORMALISED_SIZE)

        # The same socket takes another attempt; signing C for B fails and resets.
        result = attempt(ws, b, LABELS.index("C"))
        assert result["passed"] is False and result["closest"] == "C"
        assert (result["mastery"], result["streak"]) == (0, 0)

    with db_factory() as db:
        attempts = db.scalars(select(Attempt).order_by(Attempt.id)).all()
        assert [(a.user_id, a.sign_id, a.passed) for a in attempts] == [
            (user.id, b, True),
            (user.id, b, False),
        ]
        assert attempts[0].score == pytest.approx(0.9)
        progress = db.scalars(select(Progress)).one()
        assert (progress.user_id, progress.sign_id) == (user.id, b)
        assert (progress.mastery, progress.streak) == (0, 0)

    listed = signed_in.get("/api/progress").json()
    assert listed["max_mastery"] == MAX_MASTERY
    assert [(s["gloss"], s["mastery"], s["due"]) for s in listed["signs"]] == [("B", 0, True)]


def test_practice_reports_problems_without_closing(signed_in, db_factory, recogniser):
    a, z = sign_id(db_factory, "A"), sign_id(db_factory, "Z")

    def code(ws):
        reply = ws.receive_json()
        assert reply["type"] == "error"
        return reply["code"]

    with signed_in.websocket_connect("/ws/practice", headers=ORIGIN) as ws:
        ws.send_text("not json")
        assert code(ws) == "bad_message"
        ws.send_json({"type": "score"})
        assert code(ws) == "not_started"
        ws.send_json({"type": "start", "sign_id": 99999})
        assert code(ws) == "unknown_sign"
        ws.send_json({"type": "start", "sign_id": z})
        assert code(ws) == "unsupported_sign"

        ws.send_json({"type": "start", "sign_id": a})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x00" * 10)
        assert code(ws) == "bad_frame"
        ws.send_json({"type": "score"})
        assert code(ws) == "too_short"

    with db_factory() as db:
        assert db.scalars(select(Attempt)).all() == []


def test_practice_without_a_model(signed_in, db_factory):
    with signed_in.websocket_connect("/ws/practice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sign_id": sign_id(db_factory, "A")})
        assert ws.receive_json()["code"] == "no_model"


def test_score_needs_top_place_and_threshold():
    class Fixed:
        labels = LABELS

        def __init__(self, probabilities):
            self.probabilities = probabilities

        def predict(self, frames):
            return np.array(self.probabilities)

    frames = [np.frombuffer(recorded_frame(0), dtype="<f4")]
    low = score_clip(Fixed([0.5, 0.3, 0.2]), frames, "A")
    assert (low.passed, low.closest) == (False, "B")
    beaten = score_clip(Fixed([0.3, 0.1, 0.6]), frames, "A")
    assert (beaten.passed, beaten.closest) == (False, "C")
    good = score_clip(Fixed([0.7, 0.2, 0.1]), frames, "A")
    assert (good.passed, good.closest) == (True, None)


def test_review_date_moves_out_on_success_and_resets_on_failure(user, db_factory, client):
    a = sign_id(db_factory, "A")
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db_factory() as db:
        for expected in range(1, MAX_MASTERY + 2):
            progress = record_attempt(db, user.id, a, 0.9, True, at)
            mastery = min(expected, MAX_MASTERY)
            assert (progress.mastery, progress.streak) == (mastery, expected)
            assert progress.next_review_at == at + timedelta(days=REVIEW_DAYS[mastery])
        progress = record_attempt(db, user.id, a, 0.1, False, at)
        assert (progress.mastery, progress.streak, progress.next_review_at) == (0, 0, at)
        assert len(db.scalars(select(Attempt)).all()) == MAX_MASTERY + 2


def test_progress_needs_a_session(client):
    assert client.get("/api/progress").status_code == 401
