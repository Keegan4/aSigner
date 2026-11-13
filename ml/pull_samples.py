"""Pull labelled takes from the training page's Postgres database into clip files.

    python pull_samples.py --db postgresql://asigner:PASSWORD@localhost:5432/asigner
    python train.py --clips ../server/data/clips data/collected

Each take becomes data/collected/<label>/<contributor>_db<id>.npz, the same
format /record writes (raw frames, label, signer, fps), so train.py and data.py
read them unchanged. The contributor id is the anonymous one the database holds,
so train.py can still hold out a whole signer.

Running it again only downloads new takes, and removes local files for takes
that were deleted from the database since. The database URL can also come from
TRAINING_DATABASE_URL. To reach a VM's database that is not exposed on the
network, open a tunnel first: ssh -L 5432:localhost:5432 you@the-vm
"""
import argparse
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.recognition.features import FRAME_SIZE  # noqa: E402

_NAME = re.compile(r"(?P<contributor>[A-Za-z0-9]+)_db(?P<id>\d+)\.npz")
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_-]+")


def local_ids(out: Path) -> dict[int, Path]:
    found = {}
    for path in out.glob("*/*.npz"):
        match = _NAME.fullmatch(path.name)
        if match:
            found[int(match["id"])] = path
    return found


def write_clip(out: Path, sample_id: int, label: str, frames: bytes, frame_count: int,
               fps: float, contributor: str) -> Path:
    if not _SAFE_LABEL.fullmatch(label) or not contributor.isalnum():
        raise ValueError(f"Unsafe label or contributor in take {sample_id}")
    array = np.frombuffer(frames, dtype="<f4").reshape(frame_count, FRAME_SIZE)
    folder = out / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{contributor}_db{sample_id}.npz"
    pending = path.with_suffix(".part.npz")
    np.savez_compressed(pending, frames=array, label=label, signer=contributor, fps=np.float32(fps))
    os.replace(pending, path)
    return path


def sync(out: Path, remote_ids: Iterable[int], fetch) -> tuple[int, int]:
    """Make `out` match the database. `fetch(ids)` yields the rows to write as
    (id, label, frames, frame_count, fps, contributor). Returns (added, removed)."""
    remote = set(remote_ids)
    have = local_ids(out)
    removed = 0
    for sample_id, path in have.items():
        if sample_id not in remote:
            path.unlink()
            removed += 1
    missing = sorted(remote - have.keys())
    added = 0
    for row in fetch(missing) if missing else ():
        write_clip(out, *row)
        added += 1
    return added, removed


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("TRAINING_DATABASE_URL"),
                        help="Postgres URL (default: TRAINING_DATABASE_URL)")
    parser.add_argument("--out", type=Path, default=here / "data" / "collected")
    args = parser.parse_args()
    if not args.db:
        raise SystemExit("Give --db or set TRAINING_DATABASE_URL.")

    import psycopg  # only needed here, so the rest of ml/ does not require it

    url = args.db.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url) as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM training_samples")]

        def fetch(wanted: list[int]):
            # A server-side cursor streams the rows instead of loading every take at once.
            with conn.cursor(name="takes") as cur:
                cur.execute(
                    "SELECT id, label, frames, frame_count, fps, contributor "
                    "FROM training_samples WHERE id = ANY(%s) ORDER BY id",
                    (wanted,),
                )
                for sample_id, label, frames, count, fps, contributor in cur:
                    yield sample_id, label, bytes(frames), count, fps, contributor

        added, removed = sync(args.out, ids, fetch)

    per_label = Counter(path.parent.name for path in local_ids(args.out).values())
    print(f"{added} new takes, {removed} removed; {sum(per_label.values())} takes in {args.out}")
    for label, n in sorted(per_label.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
