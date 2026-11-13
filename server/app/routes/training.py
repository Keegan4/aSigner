"""The training page's API (/train): collect labelled landmark takes into the database.

No Google sign-in: access is by the shared code in TRAINING_ACCESS_CODE, sent as
the X-Training-Code header. Each browser sends its own random X-Contributor token
so its takes can be counted and undone; only a hash of it is stored.
"""
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.services import training
from app.services.training import TrainingError

router = APIRouter(prefix="/api/training", tags=["training"])


def require_access(x_training_code: str | None = Header(default=None)) -> None:
    state = training.access_state(get_settings(), x_training_code)
    if state == "closed":
        raise HTTPException(503, "Training collection is not set up on this server.")
    if state == "denied":
        raise HTTPException(403, "Wrong or missing access code.")


def contributor(x_contributor: str | None = Header(default=None)) -> str:
    try:
        return training.contributor_id(x_contributor)
    except TrainingError as e:
        raise HTTPException(e.status, str(e)) from e


def _run(fn, *args):
    try:
        return fn(*args)
    except TrainingError as e:
        raise HTTPException(e.status, str(e)) from e


class NewSign(BaseModel):
    label: str
    description: str = ""


@router.get("/signs", dependencies=[Depends(require_access)])
def signs(x_contributor: str | None = Header(default=None), db: Session = Depends(get_db)):
    """Every sign that can be recorded, with take counts (all, and this browser's)."""
    mine = None
    if x_contributor:
        mine = _run(training.contributor_id, x_contributor)
    return [vars(s) for s in training.list_signs(db, mine)]


@router.post("/signs", status_code=201, dependencies=[Depends(require_access)])
def add_sign(body: NewSign, db: Session = Depends(get_db)):
    sign = _run(training.add_sign, db, body.label, body.description)
    return {"label": sign.label, "description": sign.description}


@router.delete("/signs/{label}", status_code=204, dependencies=[Depends(require_access)])
def remove_sign(label: str, db: Session = Depends(get_db)):
    _run(training.remove_sign, db, label)


@router.post("/samples", status_code=201, dependencies=[Depends(require_access)])
async def save_sample(
    request: Request,
    label: str = Query(max_length=training.MAX_LABEL),
    fps: float = Query(gt=0, le=120),
    who: str = Depends(contributor),
    db: Session = Depends(get_db),
):
    """Store one take. The body is the raw frames back to back, 1024 bytes each."""
    body = await request.body()
    sample = _run(training.save_sample, db, label, fps, body, who)
    return {"id": sample.id, "label": sample.label, "frames": sample.frame_count}


@router.delete("/samples/{sample_id}", status_code=204, dependencies=[Depends(require_access)])
def delete_sample(sample_id: int, who: str = Depends(contributor), db: Session = Depends(get_db)):
    """Remove one of this browser's own takes, for undoing a bad one."""
    _run(training.delete_sample, db, sample_id, who)
