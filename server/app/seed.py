"""Load signs and lessons from server/seed/lessons.yaml. Safe to run repeatedly.

Run with: python -m app.seed
"""
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Lesson, LessonSign, Sign

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


def load_seed(db: Session, seed_dir: Path = SEED_DIR) -> None:
    data = yaml.safe_load((seed_dir / "lessons.yaml").read_text(encoding="utf-8"))
    for position, entry in enumerate(data["lessons"]):
        lesson = db.scalar(select(Lesson).where(Lesson.slug == entry["slug"]))
        if lesson is None:
            lesson = Lesson(slug=entry["slug"])
            db.add(lesson)
        lesson.title = entry["title"]
        lesson.summary = entry.get("summary", "")
        lesson.position = position
        lesson.items.clear()
        db.flush()
        for sign_position, raw in enumerate(entry["signs"]):
            fields = {"gloss": raw} if isinstance(raw, str) else raw
            sign = db.scalar(select(Sign).where(Sign.gloss == fields["gloss"]))
            if sign is None:
                sign = Sign(gloss=fields["gloss"])
                db.add(sign)
            sign.category = entry["slug"]
            sign.description = fields.get("description")
            sign.demo_path = fields.get("demo_path")
            sign.sign_bank_url = fields.get("sign_bank_url")
            db.flush()
            lesson.items.append(LessonSign(sign_id=sign.id, position=sign_position))
    db.commit()


if __name__ == "__main__":
    from app.core.db import SessionLocal

    with SessionLocal() as session:
        load_seed(session)
    print("Seed loaded.")
