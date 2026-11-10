"""WebSocket endpoints, one module per socket. Add a socket by adding a module
with its own `router` and including it here."""
from fastapi import APIRouter

from app.ws import echo, practice, translate

router = APIRouter()
router.include_router(echo.router)
router.include_router(practice.router)
router.include_router(translate.router)
