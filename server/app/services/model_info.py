"""What to tell learners about the model that is marking them.

A model trained on a handful of clips still loads and still gives scores, so
the screens show how much it was trained on and say plainly when that is too
little to trust.
"""
from app.recognition.recogniser import Recogniser

MIN_SIGNERS = 2  # with one signer the model has only ever seen that person's hands
MIN_TEST_CLIPS = 30  # fewer held-out clips than this and the accuracy means little


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _outside_clips(sources) -> int | None:
    """How many training clips were not recorded in the app, or None if the model does not say.

    `sources` counts clips by origin; "recorded" is the app's own recorder and
    every other key is footage converted from somewhere else.
    """
    if not isinstance(sources, dict):
        return None
    counts = [_count(n) for origin, n in sources.items() if origin != "recorded"]
    return None if None in counts else sum(counts)


def describe(recogniser: Recogniser) -> dict:
    """The model's training details, cleaned up, with `limited` set when it should not be trusted.

    `info` is optional on a recogniser and comes from a file, so nothing in it is taken on faith.
    """
    info = getattr(recogniser, "info", None)
    if not isinstance(info, dict):
        return {"known": False, "limited": True}
    clips, signers = _count(info.get("clips")), _count(info.get("signers"))
    test_clips, untested = _count(info.get("test_clips")), _count(info.get("untested_signs"))
    top1 = info.get("top1")
    top1 = float(top1) if isinstance(top1, (int, float)) and not isinstance(top1, bool) and 0 <= top1 <= 1 else None
    trained_at = info.get("trained_at")
    pretrained = info.get("pretrained")
    limited = (
        clips is None
        or signers is None
        or signers < MIN_SIGNERS
        or test_clips is None
        or test_clips < MIN_TEST_CLIPS
        or bool(untested)
    )
    return {
        "known": True,
        "clips": clips,
        "signers": signers,
        "top1": top1,
        "test_clips": test_clips,
        "untested_signs": untested,
        "trained_at": trained_at if isinstance(trained_at, str) and len(trained_at) <= 40 else None,
        "outside_clips": _outside_clips(info.get("sources")),
        "pretrained": pretrained if isinstance(pretrained, str) and len(pretrained) <= 80 else None,
        "limited": limited,
    }
