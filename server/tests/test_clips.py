import numpy as np
import pytest

from app.features import FRAME_SIZE, normalise
from app.main import app
from app.models import User
from app.routes.clips import get_clips_dir


@pytest.fixture
def clips_dir(tmp_path):
    app.dependency_overrides[get_clips_dir] = lambda: tmp_path
    return tmp_path


def clip_bytes(frames: int = 30, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (frames, FRAME_SIZE)).astype("<f4").tobytes()


def post(client, body: bytes, label: str = "A", fps: float = 30):
    return client.post("/api/clips", params={"label": label, "fps": fps}, content=body)


def test_clips_need_a_session(client, clips_dir):
    assert post(client, clip_bytes()).status_code == 401
    assert client.get("/api/clips/counts").status_code == 401


def test_saved_clip_loads_and_normalises(signed_in, user, clips_dir):
    body = clip_bytes(frames=40)
    saved = post(signed_in, body).json()
    assert saved["frames"] == 40 and saved["label"] == "A"

    with np.load(clips_dir / "A" / f"{saved['id']}.npz") as clip:
        frames = clip["frames"]
        assert frames.shape == (40, FRAME_SIZE)
        assert frames.tobytes() == body  # stored raw, exactly as received
        assert str(clip["label"]) == "A"
        assert str(clip["signer"]) == f"u{user.id}"
        assert float(clip["fps"]) == 30
    assert np.isfinite(normalise(frames[0])).all()


def test_rejects_bad_clips(signed_in, clips_dir):
    assert post(signed_in, clip_bytes(), label="NOT-A-SIGN").status_code == 404
    assert post(signed_in, clip_bytes(), label="../A").status_code == 404
    assert post(signed_in, clip_bytes()[:-1]).status_code == 400  # not whole frames
    assert post(signed_in, clip_bytes(frames=2)).status_code == 400  # too short
    assert post(signed_in, clip_bytes(frames=301)).status_code == 400  # too long
    bad = np.frombuffer(clip_bytes(), dtype="<f4").copy()
    bad[3] = np.inf
    assert post(signed_in, bad.tobytes()).status_code == 400
    assert list(clips_dir.iterdir()) == []


def test_counts_and_delete_are_per_user(signed_in, user, db_factory, clips_dir):
    first = post(signed_in, clip_bytes(seed=1)).json()["id"]
    post(signed_in, clip_bytes(seed=2))
    post(signed_in, clip_bytes(seed=3), label="B")
    assert signed_in.get("/api/clips/counts").json() == {"A": 2, "B": 1}

    with db_factory() as db:
        other = User(google_sub="sub-2", email="other@example.com")
        db.add(other)
        db.commit()
        other_id = other.id
    signed_in.post(f"/test/login/{other_id}")
    assert signed_in.get("/api/clips/counts").json() == {}
    assert signed_in.delete(f"/api/clips/A/{first}").status_code == 404

    signed_in.post(f"/test/login/{user.id}")
    assert signed_in.delete(f"/api/clips/A/{first}").status_code == 204
    assert signed_in.delete(f"/api/clips/A/{first}").status_code == 404
    assert signed_in.get("/api/clips/counts").json() == {"A": 1, "B": 1}


def test_deleting_the_account_deletes_its_clips(signed_in, user, db_factory, clips_dir):
    post(signed_in, clip_bytes(seed=1))
    post(signed_in, clip_bytes(seed=2), label="B")
    with db_factory() as db:
        other = User(google_sub="sub-2", email="other@example.com")
        db.add(other)
        db.commit()
        other_id = other.id
    signed_in.post(f"/test/login/{other_id}")
    kept = post(signed_in, clip_bytes(seed=3)).json()["id"]

    signed_in.post(f"/test/login/{user.id}")
    assert signed_in.delete("/api/me").status_code == 204
    assert [p.stem for p in clips_dir.glob("*/*.npz")] == [kept]
