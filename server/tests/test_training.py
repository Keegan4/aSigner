import numpy as np
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models import TrainingSample
from app.recognition.features import FRAME_SIZE
from app.services.training import contributor_id

ME = {"X-Contributor": "a" * 32}
OTHER = {"X-Contributor": "b" * 32}


def take(frames: int = 30, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (frames, FRAME_SIZE)).astype("<f4").tobytes()


def post(client, body: bytes, label: str = "A", fps: float = 30, headers=ME):
    return client.post(
        "/api/training/samples", params={"label": label, "fps": fps}, content=body, headers=headers
    )


def counts(client, headers=ME) -> dict[str, tuple[int, int]]:
    rows = client.get("/api/training/signs", headers=headers).json()
    return {r["label"]: (r["total"], r["mine"]) for r in rows if r["total"]}


def test_open_without_a_code_on_localhost_only(client, monkeypatch):
    assert client.get("/api/training/signs").status_code == 200
    monkeypatch.setattr(get_settings(), "frontend_url", "https://train.example.org")
    assert client.get("/api/training/signs").status_code == 503
    assert post(client, take()).status_code == 503


def test_access_code(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "training_access_code", "let-me-in")
    assert client.get("/api/training/signs").status_code == 403
    assert client.get("/api/training/signs", headers={"X-Training-Code": "nope"}).status_code == 403
    assert post(client, take()).status_code == 403
    ok = client.get("/api/training/signs", headers={"X-Training-Code": "let-me-in"})
    assert ok.status_code == 200


def test_sign_list_has_lessons_custom_and_idle(client):
    rows = client.get("/api/training/signs").json()
    assert rows[0] == {"label": "A", "description": "", "source": "lesson", "total": 0, "mine": 0}
    assert rows[-1]["label"] == "_idle" and rows[-1]["source"] == "idle"

    made = client.post("/api/training/signs", json={"label": "  thank you ", "description": "Flat hand from chin"})
    assert made.status_code == 201 and made.json()["label"] == "THANK-YOU"
    custom = [r for r in client.get("/api/training/signs").json() if r["source"] == "custom"]
    assert custom == [
        {"label": "THANK-YOU", "description": "Flat hand from chin", "source": "custom", "total": 0, "mine": 0}
    ]


@pytest.mark.parametrize("label", ["", "  ", "../x", "_idle", "café", "x" * 41, "-"])
def test_bad_sign_names_are_refused(client, label):
    assert client.post("/api/training/signs", json={"label": label}).status_code == 400


def test_duplicate_signs_are_refused(client):
    assert client.post("/api/training/signs", json={"label": "a"}).status_code == 409  # a lesson sign
    assert client.post("/api/training/signs", json={"label": "Hello"}).status_code == 201
    assert client.post("/api/training/signs", json={"label": "HELLO"}).status_code == 409


def test_saved_take_is_raw_landmarks_and_label(client, db_factory):
    body = take(frames=40)
    saved = post(client, body).json()
    assert saved["frames"] == 40 and saved["label"] == "A"
    with db_factory() as db:
        sample = db.get(TrainingSample, saved["id"])
        assert sample.frames == body
        assert sample.frame_count == 40 and sample.fps == 30
        assert sample.contributor == contributor_id(ME["X-Contributor"])
        assert ME["X-Contributor"] not in sample.contributor  # only the hash is kept


def test_bad_takes_are_refused(client, db_factory):
    assert post(client, take(), label="NOT-A-SIGN").status_code == 404
    assert post(client, take()[:-1]).status_code == 400
    assert post(client, take(frames=2)).status_code == 400
    assert post(client, take(frames=301)).status_code == 400
    bad = np.frombuffer(take(), dtype="<f4").copy()
    bad[3] = np.nan
    assert post(client, bad.tobytes()).status_code == 400
    assert post(client, take(), headers={}).status_code == 400  # no contributor token
    assert post(client, take(), headers={"X-Contributor": "short"}).status_code == 400
    with db_factory() as db:
        assert db.scalars(select(TrainingSample)).all() == []


def test_custom_and_idle_signs_can_be_recorded(client):
    client.post("/api/training/signs", json={"label": "wave"})
    assert post(client, take(), label="WAVE").status_code == 201
    assert post(client, take(), label="_idle").status_code == 201
    assert counts(client) == {"WAVE": (1, 1), "_idle": (1, 1)}


def test_counts_and_undo_are_per_contributor(client):
    first = post(client, take(seed=1)).json()["id"]
    post(client, take(seed=2))
    post(client, take(seed=3), label="B", headers=OTHER)
    assert counts(client) == {"A": (2, 2), "B": (1, 0)}
    assert counts(client, OTHER) == {"A": (2, 0), "B": (1, 1)}

    assert client.delete(f"/api/training/samples/{first}", headers=OTHER).status_code == 404
    assert client.delete(f"/api/training/samples/{first}", headers=ME).status_code == 204
    assert client.delete(f"/api/training/samples/{first}", headers=ME).status_code == 404
    assert counts(client) == {"A": (1, 1), "B": (1, 0)}


def test_only_unrecorded_custom_signs_can_be_removed(client):
    assert client.delete("/api/training/signs/A").status_code == 404  # lesson sign
    client.post("/api/training/signs", json={"label": "typo"})
    client.post("/api/training/signs", json={"label": "kept"})
    post(client, take(), label="KEPT")
    assert client.delete("/api/training/signs/KEPT").status_code == 409
    assert client.delete("/api/training/signs/TYPO").status_code == 204
    labels = {r["label"] for r in client.get("/api/training/signs").json()}
    assert "KEPT" in labels and "TYPO" not in labels
