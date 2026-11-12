import json
import math
import struct
import time

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.recognition.features import FRAME_SIZE, decode
from app.recognition.recogniser import Recogniser, UnusableModel, get_recogniser
from app.services.model_info import describe
from app.services.segmentation import Segmenter
from app.services.sentence import make_sentence
from app.services.translations import save_translation
from app.ws.practice import BAD_MODEL
from app.ws.session import authorise

router = APIRouter()

# A video frame: the 1024-byte landmark frame, then the time in the video it
# was taken from as a little-endian float64 (seconds).
VIDEO_FRAME_BYTES = FRAME_SIZE * 4 + 8


async def has_model(ws: WebSocket, recogniser: Recogniser | None) -> bool:
    """Tell the client and close the socket if there is no usable sign model."""
    if recogniser is not None and not isinstance(recogniser, UnusableModel):
        return True
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
    return False

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

    For debugging, every model prediction is also reported as {"type":
    "window", "label", "top": [[gloss, probability], ...]}, as on
    /ws/translate-video but without a time.
    """
    user_id = await authorise(ws)
    if user_id is None or not await has_model(ws, recogniser):
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
                if update.window is not None:
                    await ws.send_json(
                        {"type": "window", "label": update.seen, "top": update.window}
                    )
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


@router.websocket("/ws/translate-video")
async def translate_video(
    ws: WebSocket,
    save: bool = False,
    db: Session = Depends(get_db),
    recogniser: Recogniser | None = Depends(get_recogniser),
):
    """Translate a recorded video, frame by frame. A debugging aid for the model.

    The browser tracks the video itself (it never leaves the browser) and sends
    binary messages of VIDEO_FRAME_BYTES: a landmark frame followed by its time
    in the video. Times drive the segmenter instead of the arrival clock, so
    the result does not depend on how fast the browser can track. Times that
    do not move forward are dropped.

    Besides "gloss" and "sentence" (as /ws/translate, each with the video time
    "t"), every model prediction is reported as {"type": "window", "t", "label",
    "top": [[gloss, probability], ...]}, where label is the sign the window was
    read as (null for no sign) before the segmenter's stability check. The client sends {"type": "end"} at the
    end of the video; the sentence in progress is finished and {"type": "done"}
    follows. Sentences are stored in the user's history only with ?save=true.
    """
    user_id = await authorise(ws)
    if user_id is None or not await has_model(ws, recogniser):
        return

    segmenter = Segmenter(recogniser)
    last_time = -math.inf

    async def finish(glosses: list[str], at: float):
        if not glosses:
            return
        if save:
            sentence = await save_translation(db, user_id, glosses)
            sentence["created_at"] = sentence["created_at"].isoformat()
        else:
            text, from_model = await make_sentence(glosses)
            sentence = {"glosses": glosses, "sentence": text, "from_model": from_model}
        await ws.send_json({"type": "sentence", "t": at, **sentence})

    await ws.send_json({"type": "ready", "model": describe(recogniser)})
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("bytes")
            if payload is not None:
                if len(payload) != VIDEO_FRAME_BYTES:
                    continue
                (at,) = struct.unpack("<d", payload[-8:])
                try:
                    frame = decode(payload[:-8])
                except ValueError:
                    continue
                if not math.isfinite(at) or at <= last_time:
                    continue
                last_time = at
                update = await run_in_threadpool(segmenter.push, frame, at)
                if update.window is not None:
                    await ws.send_json(
                        {"type": "window", "t": at, "label": update.seen, "top": update.window}
                    )
                if update.gloss:
                    glosses = update.sentence if update.sentence is not None else segmenter.glosses
                    await ws.send_json(
                        {"type": "gloss", "t": at, "gloss": update.gloss, "glosses": glosses}
                    )
                if update.sentence is not None:
                    await finish(update.sentence, at)
                continue
            try:
                kind = json.loads(message.get("text") or "")["type"]
            except (ValueError, KeyError, TypeError):
                continue
            if kind == "end":
                await finish(segmenter.flush(), last_time if math.isfinite(last_time) else 0.0)
                await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
