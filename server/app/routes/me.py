from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import User
from app.routes.deps import current_user
from app.services.clips import delete_clips_for, get_clips_dir

router = APIRouter(prefix="/api/me", tags=["me"])


@router.get("")
def read_me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "avatar_url": user.avatar_url}


@router.delete("", status_code=204)
def delete_me(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    clips_dir: Path = Depends(get_clips_dir),
):
    """Remove the account and everything stored for it, including recorded clips."""
    delete_clips_for(user, clips_dir)
    db.delete(user)
    db.commit()
    request.session.clear()
