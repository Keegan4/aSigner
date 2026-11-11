from sqlalchemy import select

from app.core.config import get_settings
from app.main import app
from app.models import Sign
from app.recognition.recogniser import UnusableModel, get_recogniser
from app.services.model_info import MIN_SIGNERS, MIN_TEST_CLIPS, describe

ORIGIN = {"origin": "http://localhost:3000"}
GOOD = {
    "clips": 400,
    "signers": MIN_SIGNERS,
    "top1": 0.93,
    "test_clips": MIN_TEST_CLIPS,
    "untested_signs": 0,
    "trained_at": "2026-11-10T07:00:00+00:00",
}


class Model:
    labels = ["A"]

    def __init__(self, info=None):
        if info is not None:
            self.info = info


def test_a_well_covered_model_is_described_without_a_caution():
    assert describe(Model(GOOD)) == {
        "known": True, **GOOD, "outside_clips": None, "pretrained": None, "limited": False,
    }


def test_outside_clips_and_pretraining_are_reported():
    told = describe(Model({**GOOD, "sources": {"recorded": 380, "signbank": 12, "other": 8},
                           "pretrained": "asl-now-fingerspelling (MIT)"}))
    assert (told["outside_clips"], told["pretrained"]) == (20, "asl-now-fingerspelling (MIT)")
    assert describe(Model({**GOOD, "sources": {"recorded": 400}}))["outside_clips"] == 0
    for odd in ("signbank", {"signbank": "many"}, {"signbank": -1}):
        assert describe(Model({**GOOD, "sources": odd}))["outside_clips"] is None
    assert describe(Model({**GOOD, "pretrained": "x" * 81}))["pretrained"] is None


def test_thin_training_is_flagged():
    for change in ({"signers": 1}, {"test_clips": MIN_TEST_CLIPS - 1}, {"untested_signs": 3}):
        assert describe(Model({**GOOD, **change}))["limited"] is True


def test_missing_or_malformed_details_are_flagged_not_trusted():
    assert describe(Model()) == {"known": False, "limited": True}
    assert describe(Model("lots of clips")) == {"known": False, "limited": True}
    odd = describe(Model({"clips": "400", "signers": True, "top1": 7, "trained_at": "x" * 41}))
    assert odd == {
        "known": True,
        "clips": None,
        "signers": None,
        "top1": None,
        "test_clips": None,
        "untested_signs": None,
        "trained_at": None,
        "outside_clips": None,
        "pretrained": None,
        "limited": True,
    }


def test_practice_tells_the_learner_what_the_model_was_trained_on(signed_in, db_factory):
    app.dependency_overrides[get_recogniser] = lambda: Model({**GOOD, "signers": 1})
    with db_factory() as db:
        a = db.scalar(select(Sign.id).where(Sign.gloss == "A"))
    with signed_in.websocket_connect("/ws/practice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sign_id": a})
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    assert (ready["model"]["clips"], ready["model"]["signers"], ready["model"]["limited"]) == (400, 1, True)


def test_a_model_file_that_will_not_load_is_reported_not_raised(tmp_path):
    path = tmp_path / "signs.onnx"
    path.write_bytes(b"not a model")
    get_settings().model_path = str(path)  # restored by the default_settings fixture
    assert isinstance(get_recogniser(), UnusableModel)


def test_unusable_is_reported_whether_the_loader_raises_or_returns_none(tmp_path, monkeypatch):
    from app.recognition import onnx_recogniser

    path = tmp_path / "signs.onnx"
    path.write_bytes(b"not a model")
    get_settings().model_path = str(path)

    def raises(model_path):
        raise ValueError("stale layout")

    monkeypatch.setattr(onnx_recogniser, "load_recogniser", raises)
    assert get_recogniser().reason == "stale layout"
    monkeypatch.setattr(onnx_recogniser, "load_recogniser", lambda model_path: None)
    assert isinstance(get_recogniser(), UnusableModel)
    path.unlink()
    assert get_recogniser() is None


def test_sockets_tell_the_learner_when_the_model_needs_retraining(signed_in, db_factory):
    app.dependency_overrides[get_recogniser] = lambda: UnusableModel("old feature layout")
    with db_factory() as db:
        a = db.scalar(select(Sign.id).where(Sign.gloss == "A"))
    with signed_in.websocket_connect("/ws/practice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sign_id": a})
        reply = ws.receive_json()
        assert (reply["type"], reply["code"]) == ("error", "bad_model")
        assert "old feature layout" not in reply["message"]
        ws.send_json({"type": "score"})
        assert ws.receive_json()["code"] == "not_started"
    with signed_in.websocket_connect("/ws/translate", headers=ORIGIN) as ws:
        assert ws.receive_json()["code"] == "bad_model"
