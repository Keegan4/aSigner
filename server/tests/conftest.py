import os

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db, make_engine
from app.main import app
from app.models import User
from app.seed import load_seed


@pytest.fixture
def db_factory():
    """An empty database per test: in-memory SQLite, or TEST_DATABASE_URL if set.

    Every table in TEST_DATABASE_URL is dropped, so never point it at real data.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        engine = make_engine(url)
        Base.metadata.drop_all(engine)
    else:
        engine = make_engine("sqlite://", poolclass=StaticPool)
        # SQLite ignores ON DELETE CASCADE unless foreign keys are switched on.
        event.listen(engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def client(db_factory):
    def override_get_db():
        with db_factory() as session:
            yield session

    # Test-only route that plants a session, standing in for the Google round trip.
    if not any(getattr(r, "path", "") == "/test/login/{user_id}" for r in app.routes):

        @app.post("/test/login/{user_id}")
        def _login(user_id: int, request: Request):
            request.session["user_id"] = user_id
            return {"ok": True}

    app.dependency_overrides[get_db] = override_get_db
    with db_factory() as session:
        load_seed(session)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def user(db_factory):
    with db_factory() as session:
        u = User(google_sub="sub-1", email="learner@example.com", name="Learner")
        session.add(u)
        session.commit()
        return u


@pytest.fixture
def signed_in(client, user):
    assert client.post(f"/test/login/{user.id}").status_code == 200
    return client


@pytest.fixture(autouse=True)
def default_settings(monkeypatch, tmp_path):
    """Tests must not depend on the developer's own .env."""
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "dev_login", False)
    monkeypatch.setattr(settings, "google_client_id", "")
    monkeypatch.setattr(settings, "frontend_url", "http://localhost:3000")
    monkeypatch.setattr(settings, "model_path", str(tmp_path / "no-model" / "signs.onnx"))
    monkeypatch.setattr(settings, "clips_dir", str(tmp_path / "clips"))
