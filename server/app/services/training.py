"""Training-data collection for the training page (/train), stored in the database.

Anyone with the access code can record takes and add signs of their own. A take
is stored as its raw landmark frames and its label, plus an anonymous
contributor id; no account, name or video is involved. ml/pull_samples.py reads
the takes back out as clip files for ml/train.py.

The HTTP side is in app/routes/training.py.
"""
import hashlib
import hmac
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Sign, TrainingSample, TrainingSign
from app.recognition.features import FRAME_SIZE, IDLE_LABEL

MIN_FRAMES = 5
MAX_FRAMES = 300
FRAME_BYTES = FRAME_SIZE * 4
MAX_LABEL = 40
MAX_DESCRIPTION = 300
# A ceiling on signs added from the page, so a stuck script cannot fill the list.
MAX_CUSTOM_SIGNS = 500

# Labels become folder names in ml/, so they are kept to one safe word.
_LABEL = re.compile(r"[A-Z0-9][A-Z0-9-]*")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{16,64}")


class TrainingError(Exception):
    """A request the training page should refuse; `status` is the HTTP code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --- Access -----------------------------------------------------------------


def access_state(settings: Settings, code: str | None) -> str:
    """"open", "closed" (no code configured off localhost) or "denied"."""
    expected = settings.training_access_code
    if not expected:
        host = urlparse(settings.frontend_url).hostname
        return "open" if host in ("localhost", "127.0.0.1") else "closed"
    if code and hmac.compare_digest(code.encode(), expected.encode()):
        return "open"
    return "denied"


def contributor_id(token: str | None) -> str:
    """The stored id for a browser's random token. The token itself is never stored,
    so someone reading the database cannot delete takes as that browser."""
    if not token or not _TOKEN.fullmatch(token):
        raise TrainingError(400, "Missing or malformed contributor token.")
    return "t" + hashlib.sha256(token.encode()).hexdigest()[:16]


# --- Signs ------------------------------------------------------------------


def normalise_label(raw: str) -> str:
    """'thank you' -> 'THANK-YOU'. Raises if nothing usable is left."""
    if raw.strip().startswith("_"):  # reserved, like the "No sign" label _idle
        raise TrainingError(400, "Names starting with _ are reserved.")
    label = re.sub(r"[\s_]+", "-", raw.strip().upper())
    label = re.sub(r"-{2,}", "-", label).strip("-")
    if not label or len(label) > MAX_LABEL or not _LABEL.fullmatch(label):
        raise TrainingError(
            400, f"Use letters, digits, spaces or hyphens, up to {MAX_LABEL} characters."
        )
    return label


@dataclass
class SignCount:
    label: str
    description: str
    source: str  # "lesson", "custom" or "idle"
    total: int
    mine: int


def list_signs(db: Session, contributor: str | None) -> list[SignCount]:
    """Every collectable sign: lesson signs, then signs added here, then "No sign"."""
    totals = dict(
        db.execute(select(TrainingSample.label, func.count()).group_by(TrainingSample.label)).all()
    )
    mine: dict[str, int] = {}
    if contributor:
        mine = dict(
            db.execute(
                select(TrainingSample.label, func.count())
                .where(TrainingSample.contributor == contributor)
                .group_by(TrainingSample.label)
            ).all()
        )

    def row(label: str, description: str, source: str) -> SignCount:
        return SignCount(label, description, source, totals.get(label, 0), mine.get(label, 0))

    lesson = db.scalars(select(Sign).order_by(Sign.id)).all()
    custom = db.scalars(select(TrainingSign).order_by(TrainingSign.label)).all()
    return [
        *(row(s.gloss, s.description or "", "lesson") for s in lesson),
        *(row(s.label, s.description, "custom") for s in custom),
        row(IDLE_LABEL, "", "idle"),
    ]


def is_collectable(db: Session, label: str) -> bool:
    if label == IDLE_LABEL:
        return True
    if db.scalar(select(Sign.id).where(Sign.gloss == label)) is not None:
        return True
    return db.scalar(select(TrainingSign.id).where(TrainingSign.label == label)) is not None


def add_sign(db: Session, raw_label: str, description: str) -> TrainingSign:
    label = normalise_label(raw_label)
    description = description.strip()
    if len(description) > MAX_DESCRIPTION:
        raise TrainingError(400, f"Keep the description under {MAX_DESCRIPTION} characters.")
    if is_collectable(db, label):
        raise TrainingError(409, f"{label} is already in the list.")
    if db.scalar(select(func.count()).select_from(TrainingSign)) >= MAX_CUSTOM_SIGNS:
        raise TrainingError(409, "The sign list is full. Ask the owner to tidy it up.")
    sign = TrainingSign(label=label, description=description)
    db.add(sign)
    db.commit()
    return sign


def remove_sign(db: Session, label: str) -> None:
    """Only a sign added here, and only while nobody has recorded it, so a typo can
    be fixed without anyone being able to wipe out others' takes."""
    sign = db.scalar(select(TrainingSign).where(TrainingSign.label == label))
    if sign is None:
        raise TrainingError(404, "Only signs added on this page can be removed.")
    taken = db.scalar(select(func.count()).where(TrainingSample.label == label))
    if taken:
        raise TrainingError(409, f"{label} already has {taken} takes, so it stays.")
    db.delete(sign)
    db.commit()


# --- Samples ----------------------------------------------------------------


def decode_frames(body: bytes) -> np.ndarray:
    count, remainder = divmod(len(body), FRAME_BYTES)
    if remainder or not MIN_FRAMES <= count <= MAX_FRAMES:
        raise TrainingError(
            400, f"Expected {MIN_FRAMES}-{MAX_FRAMES} frames of {FRAME_BYTES} bytes each."
        )
    frames = np.frombuffer(body, dtype="<f4").reshape(count, FRAME_SIZE)
    if not np.isfinite(frames).all():
        raise TrainingError(400, "Take contains non-finite values.")
    return frames


def save_sample(
    db: Session, label: str, fps: float, body: bytes, contributor: str
) -> TrainingSample:
    if not is_collectable(db, label):
        raise TrainingError(404, "Unknown sign. Add it to the list first.")
    frames = decode_frames(body)
    sample = TrainingSample(
        label=label,
        frames=frames.astype("<f4").tobytes(),
        frame_count=len(frames),
        fps=fps,
        contributor=contributor,
    )
    db.add(sample)
    db.commit()
    return sample


def delete_sample(db: Session, sample_id: int, contributor: str) -> None:
    sample = db.get(TrainingSample, sample_id)
    if sample is None or sample.contributor != contributor:
        raise TrainingError(404, "Take not found.")
    db.delete(sample)
    db.commit()
