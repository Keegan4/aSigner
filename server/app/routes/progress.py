from datetime import timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Progress, Sign, User, now
from app.routes.deps import current_user
from app.services.practice import MAX_MASTERY

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("")
def list_progress(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Every sign the user has attempted, with whether it is due for review."""
    rows = db.execute(
        select(Progress, Sign)
        .join(Sign, Sign.id == Progress.sign_id)
        .where(Progress.user_id == user.id)
        .order_by(Sign.category, Sign.gloss)
    ).all()
    at = now()

    def due(progress: Progress) -> bool:
        review = progress.next_review_at
        if review is None:
            return True
        if review.tzinfo is None:  # SQLite drops the timezone
            review = review.replace(tzinfo=timezone.utc)
        return review <= at

    return {
        "max_mastery": MAX_MASTERY,
        "signs": [
            {
                "sign_id": sign.id,
                "gloss": sign.gloss,
                "category": sign.category,
                "mastery": progress.mastery,
                "streak": progress.streak,
                "next_review_at": progress.next_review_at,
                "due": due(progress),
            }
            for progress, sign in rows
        ],
    }
