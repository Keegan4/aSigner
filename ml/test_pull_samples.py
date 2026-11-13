"""pull_samples.py without a database: rows are handed to sync() directly."""
import numpy as np

from data import load_clips
from pull_samples import local_ids, sync
from train import clip_source

from app.recognition.features import FRAME_SIZE


def row(sample_id: int, label: str = "A", frames: int = 20, contributor: str = "t0123456789abcdef"):
    data = np.full((frames, FRAME_SIZE), sample_id, dtype="<f4")
    return sample_id, label, data.tobytes(), frames, 30.0, contributor


def fetcher(rows):
    by_id = {r[0]: r for r in rows}
    asked = []

    def fetch(ids):
        asked.append(list(ids))
        return [by_id[i] for i in ids]

    return fetch, asked


def test_pulled_takes_load_as_clips(tmp_path):
    rows = [row(1), row(2, "HELLO", frames=33), row(3, "_idle", contributor="tfedcba9876543210")]
    fetch, _ = fetcher(rows)
    assert sync(tmp_path, [1, 2, 3], fetch) == (3, 0)

    clips = {c.label: c for c in load_clips(tmp_path)}
    assert set(clips) == {"A", "HELLO", "_idle"}
    assert clips["HELLO"].frames.shape == (33, FRAME_SIZE)
    assert (clips["HELLO"].frames == 2).all()
    assert clips["_idle"].signer == "tfedcba9876543210"
    assert clip_source(clips["A"].signer) == "recorded"


def test_second_pull_is_incremental_and_drops_deleted_takes(tmp_path):
    fetch, asked = fetcher([row(1), row(2), row(3)])
    sync(tmp_path, [1, 2], fetch)
    assert sync(tmp_path, [2, 3], fetch) == (1, 1)
    assert asked == [[1, 2], [3]]
    assert sorted(local_ids(tmp_path)) == [2, 3]
