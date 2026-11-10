"""Where recorded training clips live on disk and who owns them.

Clips are .npz files under CLIPS_DIR, one folder per label, named
<signer>_<uuid>.npz. The HTTP side is in app/routes/clips.py.
"""
from pathlib import Path

from app.core.config import get_settings
from app.models import User


def get_clips_dir() -> Path:
    return Path(get_settings().clips_dir)


def signer_id(user: User) -> str:
    """An opaque id: clips must not carry names or email addresses."""
    return f"u{user.id}"


def delete_clips_for(user: User, clips_dir: Path) -> int:
    """Remove every clip a user recorded. Called when their account is deleted,
    which also stops a reused user id (SQLite) inheriting someone else's clips."""
    if not clips_dir.is_dir():
        return 0
    paths = list(clips_dir.glob(f"*/{signer_id(user)}_*.npz"))
    for path in paths:
        path.unlink()
    return len(paths)
