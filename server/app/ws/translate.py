import json
import time

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.recognition.features import decode
from app.recognition.recogniser import Recogniser, UnusableModel, get_recogniser
from app.services.model_info import describe
from app.services.segmentation import Segmenter
from app.services.translations import save_translation
from app.ws.practice import BAD_MODEL
from app.ws.session import authorise

router = APIRouter()

# When a frame arrived. A name of its own so tests can replace it.
clock = time.monotonic


@router.websocket("/ws/translate")
async def translate(
    ws: WebSocket,
    db: Session = Depends(get_db),
    recogniser: Recogniser | None = Depends(get_recogniser),
):
    """Translate continuous signing.

    The client streams binary landmark frames. The server answers with
    {"type": "gloss"} as each sign is recognised and {"type": "sentence"} when
    a pause ends the sentence, which is then stored. The client may send
    {"type": "flush"} to end the sentence now or {"type": "clear"} to drop it.
    """
    user_id = await authorise(ws)
    if user_id is None:
        return
    if recogniser is None or isinstance(recogniser, UnusableModel):
        missing = recogniser is None
        await ws.send_json(
            {
                "type": "error",
                "code": "no_model" if missing else "bad_model",
                "message": "Translation is not available yet: no sign model is installed."
                if missing
                else BAD_MODEL,
            }
        )
        await ws.close()
        return

    segmenter = Segmenter(recogniser)

    async def finish(glosses: list[str]):
        if glosses:
            saved = await save_translation(db, user_id, glosses)
            saved["created_at"] = saved["created_at"].isoformat()
            await ws.send_json({"type": "sentence", **saved})

    await ws.send_json({"type": "ready", "model": describe(recogniser)})
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("bytes")
            if payload is not None:
                try:
                    frame = decode(payload)
                except ValueError:
                    continue
                update = await run_in_threadpool(segmenter.push, frame, clock())
                if update.gloss:
                    glosses = update.sentence if update.sentence is not None else segmenter.glosses
                    await ws.send_json({"type": "gloss", "gloss": update.gloss, "glosses": glosses})
                if update.sentence is not None:
                    await finish(update.sentence)
                continue
            try:
                kind = json.loads(message.get("text") or "")["type"]
            except (ValueError, KeyError, TypeError):
                continue
            if kind == "flush":
                await finish(segmenter.flush())
            elif kind == "clear":
                segmenter.flush()
                await ws.send_json({"type": "cleared"})
    except WebSocketDisconnect:
        pass
