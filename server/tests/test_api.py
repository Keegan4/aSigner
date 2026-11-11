import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from app.models import Attempt, Sign, Translation, User
from app.routes.auth import upsert_user


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_protected_routes_need_a_session(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/translations").status_code == 401
    assert client.delete("/api/me").status_code == 401


def test_login_reports_missing_google_config(client):
    assert client.get("/api/auth/login", follow_redirects=False).status_code == 503


def test_me_and_logout(signed_in):
    assert signed_in.get("/api/me").json()["email"] == "learner@example.com"
    assert signed_in.post("/api/auth/logout").status_code == 204
    assert signed_in.get("/api/me").status_code == 401


def test_lessons_are_seeded_in_order(client):
    lessons = client.get("/api/lessons").json()
    assert lessons[0]["slug"] == "alphabet"
    assert lessons[0]["sign_count"] == 26
    signs = client.get("/api/lessons/alphabet").json()["signs"]
    assert [s["gloss"] for s in signs] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert client.get("/api/lessons/nope").status_code == 404


def test_basic_words_link_to_the_sign_bank_and_describe_nothing(client):
    lessons = client.get("/api/lessons").json()
    assert [l["slug"] for l in lessons] == ["alphabet", "basic-words"]
    signs = client.get("/api/lessons/basic-words").json()["signs"]
    assert len(signs) == lessons[1]["sign_count"] == 24
    assert {"YES", "NO"} <= {s["gloss"] for s in signs}  # not read as YAML booleans
    for sign in signs:
        assert sign["sign_bank_url"].startswith("https://blogs.ntu.edu.sg/sgslsignbank/word/")
        assert sign["description"] is None and sign["demo_path"] is None


def test_upsert_user_updates_instead_of_duplicating(db_factory):
    with db_factory() as db:
        first = upsert_user(db, {"sub": "g-1", "email": "a@example.com", "name": "A"})
        second = upsert_user(db, {"sub": "g-1", "email": "b@example.com", "name": "B"})
        assert first.id == second.id
        assert db.scalars(select(User)).one().email == "b@example.com"


def test_translation_history_is_per_user(signed_in, user, db_factory):
    with db_factory() as db:
        other = User(google_sub="sub-2", email="other@example.com")
        db.add(other)
        db.flush()
        mine = Translation(user_id=user.id, glosses=["ME", "HOME", "GO"], sentence="I am going home.")
        theirs = Translation(user_id=other.id, glosses=["HELLO"], sentence="Hello.")
        db.add_all([mine, theirs])
        db.commit()
        mine_id, theirs_id = mine.id, theirs.id

    rows = signed_in.get("/api/translations").json()
    assert [r["sentence"] for r in rows] == ["I am going home."]
    assert rows[0]["glosses"] == ["ME", "HOME", "GO"]
    assert signed_in.delete(f"/api/translations/{theirs_id}").status_code == 404
    assert signed_in.delete(f"/api/translations/{mine_id}").status_code == 204
    assert signed_in.get("/api/translations").json() == []


def test_delete_me_removes_all_user_rows(signed_in, user, db_factory):
    with db_factory() as db:
        sign = db.scalars(select(Sign)).first()
        db.add(Translation(user_id=user.id, glosses=[], sentence="x"))
        db.add(Attempt(user_id=user.id, sign_id=sign.id, score=0.9, passed=True))
        db.commit()

    assert signed_in.delete("/api/me").status_code == 204
    assert signed_in.get("/api/me").status_code == 401
    with db_factory() as db:
        assert db.scalars(select(User)).all() == []
        assert db.scalars(select(Translation)).all() == []
        assert db.scalars(select(Attempt)).all() == []


ORIGIN = {"origin": "http://localhost:3000"}


def test_ws_rejects_anonymous(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/echo", headers=ORIGIN) as ws:
            ws.receive_json()
    assert exc.value.code == 4401


def test_ws_rejects_other_origins(signed_in):
    with pytest.raises(WebSocketDisconnect) as exc:
        with signed_in.websocket_connect("/ws/echo", headers={"origin": "https://evil.example"}):
            pass
    assert exc.value.code == 4403


def test_ws_echo_counts_frames(signed_in):
    with signed_in.websocket_connect("/ws/echo", headers=ORIGIN) as ws:
        ws.send_bytes(b"\x00" * 2048)
        assert ws.receive_json() == {"frames": 1, "bytes": 2048}
        ws.send_bytes(b"\x00" * 10)
        assert ws.receive_json() == {"frames": 2, "bytes": 10}


def test_dev_login_is_off_by_default(client):
    assert client.get("/api/auth/dev-login", follow_redirects=False).status_code == 404
    assert client.get("/api/auth/methods").json() == {"google": False, "dev": False}


def test_dev_login_needs_flag_and_localhost(client, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "dev_login", True)
    monkeypatch.setattr(settings, "frontend_url", "https://asigner.example")
    assert client.get("/api/auth/dev-login", follow_redirects=False).status_code == 404

    monkeypatch.setattr(settings, "frontend_url", "http://localhost:3000")
    assert client.get("/api/auth/dev-login", follow_redirects=False).status_code == 307
    assert client.get("/api/me").json()["email"] == "dev@localhost"
