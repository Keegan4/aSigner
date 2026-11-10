from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import User

router = APIRouter(prefix="/api/me", tags=["me"])


@router.get("")
def read_me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "avatar_url": user.avatar_url}


@router.delete("", status_code=204)
def delete_me(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Remove the account and everything stored for it."""
    db.delete(user)
    db.commit()
    request.session.clear()
