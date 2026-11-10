from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Lesson

router = APIRouter(prefix="/api/lessons", tags=["lessons"])


@router.get("")
def list_lessons(db: Session = Depends(get_db)):
    lessons = db.scalars(select(Lesson).order_by(Lesson.position)).all()
    return [
        {
            "slug": lesson.slug,
            "title": lesson.title,
            "summary": lesson.summary,
            "sign_count": len(lesson.items),
        }
        for lesson in lessons
    ]


@router.get("/{slug}")
def read_lesson(slug: str, db: Session = Depends(get_db)):
    lesson = db.scalar(select(Lesson).where(Lesson.slug == slug))
    if lesson is None:
        raise HTTPException(404, "Lesson not found")
    return {
        "slug": lesson.slug,
        "title": lesson.title,
        "summary": lesson.summary,
        "signs": [
            {
                "id": item.sign.id,
                "gloss": item.sign.gloss,
                "description": item.sign.description,
                "demo_path": item.sign.demo_path,
                "sign_bank_url": item.sign.sign_bank_url,
            }
            for item in lesson.items
        ],
    }
