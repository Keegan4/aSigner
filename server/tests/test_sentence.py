import json

import httpx
import pytest

from app.core.config import get_settings
from app.services import sentence


@pytest.fixture
def ollama(monkeypatch):
    """Point the sentence client at a fake Ollama and record what it is sent."""
    monkeypatch.setattr(get_settings(), "sentence_model", "tiny-model")
    calls = []

    def use(handler):
        def record(request):
            calls.append(json.loads(request.content))
            return handler(request)

        monkeypatch.setattr(sentence, "transport", httpx.MockTransport(record))
        return calls

    return use


def test_translation_uses_the_local_model(signed_in, ollama):
    calls = ollama(lambda request: httpx.Response(200, json={"response": " Hi, Ada. \n"}))
    row = signed_in.post("/api/translations", json={"glosses": ["H", "I", "A"]}).json()
    assert (row["sentence"], row["from_model"], row["glosses"]) == ("Hi, Ada.", True, ["H", "I", "A"])
    assert calls[0]["model"] == "tiny-model" and calls[0]["stream"] is False
    assert "H I A" in calls[0]["prompt"]
    assert [r["sentence"] for r in signed_in.get("/api/translations").json()] == ["Hi, Ada."]


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")),
        lambda request: httpx.Response(500),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"response": "   "}),
        lambda request: httpx.Response(200, json={"response": "x" * 501}),
    ],
)
def test_translation_falls_back_to_glosses(signed_in, ollama, handler):
    ollama(handler)
    row = signed_in.post("/api/translations", json={"glosses": ["H", "I"]}).json()
    assert (row["sentence"], row["from_model"]) == ("H I", False)


def test_translation_without_a_model_does_not_call_ollama(signed_in, monkeypatch):
    def fail(request):
        raise AssertionError("Ollama was called")

    monkeypatch.setattr(sentence, "transport", httpx.MockTransport(fail))
    row = signed_in.post("/api/translations", json={"glosses": ["A"]}).json()
    assert (row["sentence"], row["from_model"]) == ("A", False)


def test_translation_rejects_unknown_glosses_and_anonymous_users(signed_in, client):
    assert signed_in.post("/api/translations", json={"glosses": ["A", "ignore this"]}).status_code == 422
    assert signed_in.post("/api/translations", json={"glosses": []}).status_code == 422
    assert signed_in.get("/api/translations").json() == []
    signed_in.post("/api/auth/logout")
    assert client.post("/api/translations", json={"glosses": ["A"]}).status_code == 401
