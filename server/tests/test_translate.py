import numpy as np
import pytest
from sqlalchemy import select

from app.main import app
from app.models import Translation
from app.recognition.features import (
    FRAME_SIZE,
    HAND_POINTS,
    IDLE_LABEL,
    POSE,
    RIGHT_HAND,
)
from app.recognition.recogniser import get_recogniser
from app.services.segmentation import (
    MAX_GLOSSES,
    SENTENCE_GAP_SECONDS,
    STABLE_PREDICTIONS,
    STRIDE_SECONDS,
    Segmenter,
)
from app.ws import translate as ws_module

ORIGIN = {"origin": "http://localhost:3000"}
LABELS = ["A", "B", "C", IDLE_LABEL]
FPS = 20


class FakeRecogniser:
    """Names the sign from the newest frame: the right index fingertip's x picks the label."""

    labels = LABELS

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def predict(self, frames):
        marker = frames[-1, HAND_POINTS * 3 + 8 * 3]
        out = np.full(len(LABELS), (1 - self.confidence) / (len(LABELS) - 1))
        out[int(round(marker))] = self.confidence
        return out


def frame(label: str | None) -> np.ndarray:
    """A raw frame showing `label`, or no hands at all for None."""
    raw = np.zeros(FRAME_SIZE, dtype="<f4")
    pose = raw[POSE : POSE + 75].reshape(25, 3)
    pose[11], pose[12] = (0.6, 0.5, 0), (0.4, 0.5, 0)
    if label is not None:
        hand = raw[RIGHT_HAND : RIGHT_HAND + 63].reshape(21, 3)
        hand[0], hand[9] = (0.5, 0.5, 0), (0.5, 0.4, 0)
        hand[8] = (0.5 + 0.1 * LABELS.index(label), 0.5, 0)
    return raw


class Signer:
    """Feeds a segmenter frames at a steady rate and collects what it reports."""

    def __init__(self, recogniser=None):
        self.segmenter = Segmenter(recogniser or FakeRecogniser())
        self.at = 0.0
        self.glosses, self.sentences = [], []

    def sign(self, label, seconds):
        for _ in range(int(seconds * FPS)):
            self.at += 1 / FPS
            update = self.segmenter.push(frame(label), self.at)
            if update.gloss:
                self.glosses.append(update.gloss)
            if update.sentence is not None:
                self.sentences.append(update.sentence)
        return self


def test_held_signs_are_counted_once_each():
    s = Signer().sign("A", 3).sign("B", 2).sign("C", 2)
    assert s.glosses == ["A", "B", "C"]
    assert s.sentences == []


def test_a_pause_lets_a_sign_repeat_and_a_long_pause_ends_the_sentence():
    s = Signer().sign("A", 2).sign(IDLE_LABEL, 1).sign("A", 2)
    assert s.glosses == ["A", "A"] and s.sentences == []
    s.sign(None, SENTENCE_GAP_SECONDS + 1)
    assert s.sentences == [["A", "A"]]
    assert s.segmenter.glosses == []
    # Nothing more is reported while nobody is signing.
    s.sign(None, 5)
    assert s.sentences == [["A", "A"]]


def test_brief_or_unsure_signs_are_ignored():
    # Shorter than the model needs to name the sign several times running.
    assert Signer().sign(None, 2).sign("B", STRIDE_SECONDS * (STABLE_PREDICTIONS - 1)).sign(None, 2).glosses == []
    assert Signer(FakeRecogniser(confidence=0.5)).sign("A", 4).glosses == []


def test_a_very_long_sentence_is_cut():
    s = Signer()
    for _ in range(MAX_GLOSSES):
        s.sign("A", 1.5).sign(IDLE_LABEL, 1)
    assert len(s.sentences) == 1 and len(s.sentences[0]) == MAX_GLOSSES
    assert s.segmenter.glosses == []


@pytest.fixture
def clock(monkeypatch):
    """Each frame the socket receives arrives 1/FPS after the one before."""
    now = [0.0]

    def tick():
        now[0] += 1 / FPS
        return now[0]

    monkeypatch.setattr(ws_module, "clock", tick)


def test_translate_socket_stores_the_sentence(signed_in, user, db_factory, clock):
    app.dependency_overrides[get_recogniser] = lambda: FakeRecogniser()
    with signed_in.websocket_connect("/ws/translate", headers=ORIGIN) as ws:
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(b"\x00" * 10)  # a malformed frame is dropped
        for label, seconds in (("A", 2), ("B", 2)):
            for _ in range(seconds * FPS):
                ws.send_bytes(frame(label).tobytes())
        assert ws.receive_json() == {"type": "gloss", "gloss": "A", "glosses": ["A"]}
        assert ws.receive_json() == {"type": "gloss", "gloss": "B", "glosses": ["A", "B"]}
        ws.send_json({"type": "flush"})
        sentence = ws.receive_json()
        assert (sentence["type"], sentence["glosses"], sentence["sentence"]) == ("sentence", ["A", "B"], "A B")
        assert sentence["from_model"] is False

        for _ in range(2 * FPS):
            ws.send_bytes(frame("C").tobytes())
        assert ws.receive_json()["gloss"] == "C"
        ws.send_json({"type": "clear"})
        assert ws.receive_json() == {"type": "cleared"}
        ws.send_json({"type": "flush"})  # nothing left to store

    with db_factory() as db:
        rows = db.scalars(select(Translation)).all()
        assert [(r.user_id, r.glosses, r.sentence) for r in rows] == [(user.id, ["A", "B"], "A B")]
    assert [r["id"] for r in signed_in.get("/api/translations").json()] == [sentence["id"]]


def test_translate_socket_without_a_model(signed_in):
    with signed_in.websocket_connect("/ws/translate", headers=ORIGIN) as ws:
        assert ws.receive_json()["code"] == "no_model"
