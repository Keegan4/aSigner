import json

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .features import decode
from .models import Sign
from .practice import MAX_FRAMES, MIN_FRAMES, record_attempt, score_clip
from .recogniser import Recogniser, get_recogniser

router = APIRouter()


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


@router.websocket("/ws/echo")
async def echo(ws: WebSocket):
    """Plumbing check: counts landmark frames and reports their size."""
    if await authorise(ws) is None:
        return
    frames = 0
    try:
        while True:
            data = await ws.receive_bytes()
            frames += 1
            await ws.send_json({"frames": frames, "bytes": len(data)})
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/practice")
async def practice(
    ws: WebSocket,
    db: Session = Depends(get_db),
    recogniser: Recogniser | None = Depends(get_recogniser),
):
    """Score attempts at a named sign.

    Text messages are JSON. The client sends {"type": "start", "sign_id": n},
    then binary landmark frames, then {"type": "score"}. The server answers
    with "ready", "result" or "error". A result stores the attempt and updates
    progress; the client may then start again on the same socket.
    """
    user_id = await authorise(ws)
    if user_id is None:
        return

    async def error(code: str, message: str):
        await ws.send_json({"type": "error", "code": code, "message": message})

    sign: Sign | None = None
    frames: list[np.ndarray] = []
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("bytes")
            if payload is not None:
                if sign is None:
                    continue  # frames outside an attempt are ignored
                try:
                    frames.append(decode(payload))
                except ValueError as exc:
                    await error("bad_frame", str(exc))
                    continue
                del frames[:-MAX_FRAMES]
                continue

            try:
                command = json.loads(message.get("text") or "")
                kind = command["type"]
            except (ValueError, KeyError, TypeError):
                await error("bad_message", "Expected a JSON object with a type.")
                continue

            if kind == "start":
                sign_id = command.get("sign_id")
                target = db.get(Sign, sign_id) if isinstance(sign_id, int) else None
                if target is None:
                    await error("unknown_sign", "That sign does not exist.")
                elif recogniser is None:
                    await error("no_model", "Scoring is not available yet: no sign model is installed.")
                elif target.gloss not in recogniser.labels:
                    await error("unsupported_sign", "The sign model has not learnt this sign yet.")
                else:
                    sign, frames = target, []
                    await ws.send_json({"type": "ready", "sign_id": sign.id, "gloss": sign.gloss})
            elif kind == "score":
                if sign is None or recogniser is None:
                    await error("not_started", "Send start before score.")
                elif len(frames) < MIN_FRAMES:
                    await error("too_short", "Not enough frames were received to score.")
                else:
                    result = await run_in_threadpool(score_clip, recogniser, frames, sign.gloss)
                    progress = record_attempt(db, user_id, sign.id, result.score, result.passed)
                    await ws.send_json(
                        {
                            "type": "result",
                            "sign_id": sign.id,
                            "score": result.score,
                            "passed": result.passed,
                            "closest": result.closest,
                            "mastery": progress.mastery,
                            "streak": progress.streak,
                            "next_review_at": progress.next_review_at.isoformat(),
                        }
                    )
                    sign, frames = None, []
            else:
                await error("bad_message", "Unknown message type.")
    except WebSocketDisconnect:
        pass
