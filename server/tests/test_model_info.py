from sqlalchemy import select

from app.main import app
from app.models import Sign
from app.recognition.recogniser import get_recogniser
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
    assert describe(Model(GOOD)) == {"known": True, **GOOD, "limited": False}


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
