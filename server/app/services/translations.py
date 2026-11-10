"""Storing a finished translation. Used by the REST route and the live socket."""
from sqlalchemy.orm import Session

from app.models import Translation
from app.services.sentence import make_sentence


async def save_translation(db: Session, user_id: int, glosses: list[str]) -> dict:
    """Write the sentence for `glosses`, store it and return it as the API shows it."""
    sentence, from_model = await make_sentence(glosses)
    row = Translation(user_id=user_id, glosses=glosses, sentence=sentence)
    db.add(row)
    db.commit()
    return {
        "id": row.id,
        "glosses": row.glosses,
        "sentence": row.sentence,
        "created_at": row.created_at,
        "from_model": from_model,
    }
