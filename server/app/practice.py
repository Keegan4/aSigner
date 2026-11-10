"""Scoring one practice attempt and scheduling the next review."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .features import IDLE_LABEL, normalise
from .models import Attempt, Progress, now
from .recogniser import Recogniser

# Used unless the recogniser carries its own `pass_score` (chosen from the
# false-accept / false-reject curve when the model is trained).
PASS_SCORE = 0.6
MIN_FRAMES = 5
MAX_FRAMES = 300  # about ten seconds; older frames are dropped

# Days until the next review, indexed by mastery after a successful attempt.
REVIEW_DAYS = (0, 1, 3, 7, 14, 30)
MAX_MASTERY = len(REVIEW_DAYS) - 1


@dataclass
class Score:
    score: float
    passed: bool
    closest: str | None
    """The sign the model thought more likely than the target, if any."""
    no_sign: bool = False
    """The model saw no sign at all, rather than a different one."""


def pass_score(recogniser: Recogniser) -> float:
    return getattr(recogniser, "pass_score", None) or PASS_SCORE


def score_clip(recogniser: Recogniser, frames: list[np.ndarray], target: str) -> Score:
    """Score raw frames against the target gloss, which must be one of the model's labels."""
    clip = np.stack([normalise(frame) for frame in frames])
    probabilities = np.asarray(recogniser.predict(clip), dtype=np.float64)
    index = recogniser.labels.index(target)
    score = float(probabilities[index])
    best = int(probabilities.argmax())
    passed = best == index and score >= pass_score(recogniser)
    if best == index:
        others = probabilities.copy()
        others[index] = -1.0
        best = int(others.argmax())
    closest = None if passed or len(probabilities) < 2 else recogniser.labels[best]
    if closest == IDLE_LABEL:
        return Score(score=score, passed=False, closest=None, no_sign=True)
    return Score(score=score, passed=passed, closest=closest)


def record_attempt(
    db: Session, user_id: int, sign_id: int, score: float, passed: bool, at: datetime | None = None
) -> Progress:
    """Store the attempt and move the sign's review date: out on success, back to now on failure."""
    at = at or now()
    db.add(Attempt(user_id=user_id, sign_id=sign_id, score=score, passed=passed, created_at=at))
    progress = db.scalar(
        select(Progress).where(Progress.user_id == user_id, Progress.sign_id == sign_id)
    )
    if progress is None:
        progress = Progress(user_id=user_id, sign_id=sign_id, mastery=0, streak=0)
        db.add(progress)
    if passed:
        progress.streak += 1
        progress.mastery = min(progress.mastery + 1, MAX_MASTERY)
        progress.next_review_at = at + timedelta(days=REVIEW_DAYS[progress.mastery])
    else:
        progress.streak = 0
        progress.mastery = 0
        progress.next_review_at = at
    db.commit()
    return progress
