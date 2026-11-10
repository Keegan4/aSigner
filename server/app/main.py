from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from . import auth, ws
from .config import get_settings
from .routes import lessons, me, progress, translations

settings = get_settings()

app = FastAPI(title="aSigner API")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="asigner_session",
    same_site="lax",
    https_only=settings.frontend_url.startswith("https://"),
    max_age=60 * 60 * 24 * 14,
)

app.include_router(auth.router)
app.include_router(me.router)
app.include_router(lessons.router)
app.include_router(translations.router)
app.include_router(progress.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
