from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import User


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user
