from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.session import authorise

router = APIRouter()


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
