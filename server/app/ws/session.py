"""Who is on the other end of a WebSocket."""
from fastapi import WebSocket

from app.core.config import get_settings


async def authorise(ws: WebSocket) -> int | None:
    """Accept the socket and return the signed-in user id, or close it and return None.

    Browsers attach cookies to WebSocket handshakes from any page, so the
    Origin header is checked to stop other sites riding the user's session.
    A foreign origin is refused before the handshake. A missing session is
    accepted and then closed, because a browser only sees the close code
    (4401, which the client turns into a sign-in prompt) after accepting.
    """
    if ws.headers.get("origin") != get_settings().frontend_url:
        await ws.close(code=4403, reason="Origin not allowed")
        return None
    await ws.accept()
    user_id = ws.session.get("user_id")
    if not user_id:
        await ws.close(code=4401, reason="Not signed in")
        return None
    return user_id
