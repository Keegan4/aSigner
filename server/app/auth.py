from urllib.parse import urlparse

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=get_settings().google_client_id,
    client_secret=get_settings().google_client_secret,
    client_kwargs={"scope": "openid email profile"},
)


def upsert_user(db: Session, info: dict) -> User:
    user = db.scalar(select(User).where(User.google_sub == info["sub"]))
    if user is None:
        user = User(google_sub=info["sub"])
        db.add(user)
    user.email = info.get("email", "")
    user.name = info.get("name", "")
    user.avatar_url = info.get("picture")
    db.commit()
    return user


@router.get("/login")
async def login(request: Request):
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(503, "Google sign-in is not configured (set GOOGLE_CLIENT_ID).")
    # Built from FRONTEND_URL so it matches the URI registered with Google,
    # whatever proxy sits in front of this server.
    redirect_uri = f"{settings.frontend_url}/api/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        raise HTTPException(400, f"Google sign-in failed: {exc.error}") from exc
    info = token.get("userinfo")
    if not info or not info.get("email_verified", False):
        raise HTTPException(400, "Google did not return a verified account.")
    user = upsert_user(db, info)
    request.session.clear()
    request.session["user_id"] = user.id
    return RedirectResponse(get_settings().frontend_url)


def dev_login_allowed() -> bool:
    settings = get_settings()
    host = urlparse(settings.frontend_url).hostname
    return settings.dev_login and host in ("localhost", "127.0.0.1")


@router.get("/dev-login")
def dev_login(request: Request, db: Session = Depends(get_db)):
    """Sign in as a fixed local user without Google. Off unless DEV_LOGIN=true on localhost."""
    if not dev_login_allowed():
        raise HTTPException(404, "Not found")
    user = upsert_user(db, {"sub": "dev-local", "email": "dev@localhost", "name": "Local developer"})
    request.session.clear()
    request.session["user_id"] = user.id
    return RedirectResponse(get_settings().frontend_url)


@router.get("/methods")
def methods():
    """Which sign-in options the web app should offer."""
    return {"google": bool(get_settings().google_client_id), "dev": dev_login_allowed()}


@router.post("/logout", status_code=204)
def logout(request: Request):
    request.session.clear()
