from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from app import routes, ws
from app.core.config import get_settings

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

app.include_router(routes.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
