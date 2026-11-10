from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import Translation, User

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


@router.delete("/{translation_id}", status_code=204)
def delete_translation(
    translation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    row = db.get(Translation, translation_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(404, "Translation not found")
    db.delete(row)
    db.commit()
