from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Sign, Translation, User
from app.routes.deps import current_user
from app.services.translations import save_translation

router = APIRouter(prefix="/api/translations", tags=["translations"])


@router.get("")
def list_translations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Translation)
        .where(Translation.user_id == user.id)
        .order_by(Translation.created_at.desc())
        .limit(200)
    ).all()
    return [
        {"id": t.id, "glosses": t.glosses, "sentence": t.sentence, "created_at": t.created_at}
        for t in rows
    ]


class NewTranslation(BaseModel):
    glosses: list[str] = Field(min_length=1, max_length=50)


@router.post("", status_code=201)
async def create_translation(
    body: NewTranslation, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Store a run of recognised glosses with its English sentence.

    Only glosses of known signs are accepted, so free text never reaches the
    sentence model.
    """
    known = set(db.scalars(select(Sign.gloss).where(Sign.gloss.in_(body.glosses))))
    if not known.issuperset(body.glosses):
        raise HTTPException(422, "Every gloss must be a known sign.")
    return await save_translation(db, user.id, body.glosses)


@router.delete("/{translation_id}", status_code=204)
def delete_translation(
    translation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    row = db.get(Translation, translation_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(404, "Translation not found")
    db.delete(row)
    db.commit()
