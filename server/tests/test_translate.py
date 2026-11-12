import struct

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
        assert ws.receive_json() == {"type": "ready", "model": {"known": False, "limited": True}}
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


def video_frame(label: str | None, at: float) -> bytes:
    return frame(label).tobytes() + struct.pack("<d", at)


def test_segmenter_reports_each_window():
    segmenter = Segmenter(FakeRecogniser())
    windows = []
    for i in range(1, 3 * FPS):
        update = segmenter.push(frame("A"), i / FPS)
        if update.window is not None:
            windows.append((update.seen, update.window))
    assert windows[0] == (None, [])  # too little seen yet
    seen, top = windows[-1]
    assert seen == "A" and top[0] == ("A", pytest.approx(0.9)) and len(top) == 3


def test_translate_video_socket_uses_video_time(signed_in, db_factory):
    app.dependency_overrides[get_recogniser] = lambda: FakeRecogniser()
    # No clock fixture: the frames arrive at once, but carry video times.
    with signed_in.websocket_connect("/ws/translate-video", headers=ORIGIN) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(frame("A").tobytes())  # no time: dropped
        at = 0.0
        for label, seconds in (("A", 2), ("B", 2)):
            for _ in range(seconds * FPS):
                at += 1 / FPS
                ws.send_bytes(video_frame(label, at))
        ws.send_bytes(video_frame("C", at))  # time does not move forward: dropped
        ws.send_json({"type": "end"})
        messages = []
        while (message := ws.receive_json())["type"] != "done":
            messages.append(message)

    windows = [m for m in messages if m["type"] == "window"]
    assert windows and all(b["t"] > a["t"] for a, b in zip(windows, windows[1:]))
    assert windows[-1]["label"] == "B" and windows[-1]["top"][0][0] == "B"
    glosses = [m for m in messages if m["type"] == "gloss"]
    assert [g["gloss"] for g in glosses] == ["A", "B"]
    assert 0 < glosses[0]["t"] < 2 < glosses[1]["t"] <= at
    sentence = messages[-1]
    assert (sentence["type"], sentence["glosses"], sentence["sentence"]) == ("sentence", ["A", "B"], "A B")
    assert "id" not in sentence
    with db_factory() as db:
        assert db.scalars(select(Translation)).all() == []  # not saved unless asked


def test_translate_video_socket_can_save(signed_in, user, db_factory):
    app.dependency_overrides[get_recogniser] = lambda: FakeRecogniser()
    with signed_in.websocket_connect("/ws/translate-video?save=true", headers=ORIGIN) as ws:
        ws.receive_json()
        for i in range(1, 2 * FPS):
            ws.send_bytes(video_frame("C", i / FPS))
        ws.send_json({"type": "end"})
        while (message := ws.receive_json())["type"] != "sentence":
            pass
        assert ws.receive_json() == {"type": "done"}
    with db_factory() as db:
        rows = db.scalars(select(Translation)).all()
        assert [(r.user_id, r.glosses, r.id) for r in rows] == [(user.id, ["C"], message["id"])]


def test_translate_video_socket_without_a_model(signed_in):
    with signed_in.websocket_connect("/ws/translate-video", headers=ORIGIN) as ws:
        assert ws.receive_json()["code"] == "no_model"
