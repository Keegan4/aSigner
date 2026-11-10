"""Training-clip collection (handoff task T1).

A clip is a short run of raw landmark frames with the sign it shows. Clips are
written under CLIPS_DIR as one .npz each and read by the training code in ml/.
No video is involved at any point.
"""
import re
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import current_user
from ..features import FRAME_SIZE
from ..models import Sign, User

router = APIRouter(prefix="/api/clips", tags=["clips"])

MIN_FRAMES = 5
MAX_FRAMES = 300
_FRAME_BYTES = FRAME_SIZE * 4
_SAFE = re.compile(r"[A-Za-z0-9_-]+")


def get_clips_dir() -> Path:
    return Path(get_settings().clips_dir)


def signer_id(user: User) -> str:
    """An opaque id: clips must not carry names or email addresses."""
    return f"u{user.id}"


@router.post("", status_code=201)
async def save_clip(
    request: Request,
    label: str = Query(max_length=80),
    fps: float = Query(gt=0, le=120),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    clips_dir: Path = Depends(get_clips_dir),
):
    """Store one clip. The body is the raw frames back to back, 1024 bytes each."""
    if not _SAFE.fullmatch(label) or db.scalar(select(Sign).where(Sign.gloss == label)) is None:
        raise HTTPException(404, "Unknown sign.")
    body = await request.body()
    count, remainder = divmod(len(body), _FRAME_BYTES)
    if remainder or not MIN_FRAMES <= count <= MAX_FRAMES:
        raise HTTPException(
            400, f"Expected {MIN_FRAMES}-{MAX_FRAMES} frames of {_FRAME_BYTES} bytes each."
        )
    frames = np.frombuffer(body, dtype="<f4").reshape(count, FRAME_SIZE)
    if not np.isfinite(frames).all():
        raise HTTPException(400, "Clip contains non-finite values.")

    signer = signer_id(user)
    clip_id = f"{signer}_{uuid.uuid4().hex}"
    folder = clips_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{clip_id}.npz", frames=frames, label=label, signer=signer, fps=np.float32(fps)
    )
    return {"id": clip_id, "label": label, "frames": count}


@router.get("/counts")
def clip_counts(user: User = Depends(current_user), clips_dir: Path = Depends(get_clips_dir)):
    """How many clips this user has recorded for each sign."""
    prefix = f"{signer_id(user)}_"
    if not clips_dir.is_dir():
        return {}
    counts = {
        folder.name: sum(1 for _ in folder.glob(f"{prefix}*.npz"))
        for folder in clips_dir.iterdir()
        if folder.is_dir()
    }
    return {label: n for label, n in counts.items() if n}


@router.delete("/{label}/{clip_id}", status_code=204)
def delete_clip(
    label: str,
    clip_id: str,
    user: User = Depends(current_user),
    clips_dir: Path = Depends(get_clips_dir),
):
    """Remove one of the user's own clips, for undoing a bad take."""
    owned = clip_id.startswith(f"{signer_id(user)}_")
    if not (owned and _SAFE.fullmatch(label) and _SAFE.fullmatch(clip_id)):
        raise HTTPException(404, "Clip not found.")
    path = clips_dir / label / f"{clip_id}.npz"
    if not path.is_file():
        raise HTTPException(404, "Clip not found.")
    path.unlink()
