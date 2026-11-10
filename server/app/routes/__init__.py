"""HTTP endpoints, one module per resource. Add a resource by adding a module
with its own `router` and including it here."""
from fastapi import APIRouter

from app.routes import auth, clips, lessons, me, progress, translations

router = APIRouter()
for module in (auth, me, lessons, translations, progress, clips):
    router.include_router(module.router)
