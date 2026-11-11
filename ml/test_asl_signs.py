"""Checks the isolated-sign loader and pretraining on it, on made-up files. Says nothing about real accuracy."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from asl_signs import LABEL_PREFIX, SIGNER_PREFIX, SOURCE, load_sequences, sequence_to_clip
from pretrain import pretrain
from test_pretrain import write_samples

import public_data
from app.recognition.features import FACE, FRAME_SIZE, IDLE_LABEL, LEFT_HAND, POSE, RIGHT_HAND

COUNTS = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}


def write_sequence(path, parts, index_name="landmark_index"):
    """`parts` maps a part name to (time, points, 3); NaN rows are how the dataset marks a hand out of view."""
    frames = len(next(iter(parts.values())))
    rows = {"frame": [], "row_id": [], "type": [], index_name: [], "x": [], "y": [], "z": []}
    for t in range(frames):
        for name, count in COUNTS.items():
            values = parts.get(name, np.full((frames, count, 3), np.nan))[t]
            for i in range(count):
                rows["frame"].append(20 + t)  # frame numbers do not start at zero in the real files
                rows["row_id"].append(f"{20 + t}-{name}-{i}")
                rows["type"].append(name)
                rows[index_name].append(i)
                for axis, value in zip("xyz", values[i]):
                    rows[axis].append(float(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), path)


def write_dataset(folder, signs=("hello", "dog", "TV"), signers=("11", "22", "33"), per_signer=3, frames=10):
    lines = ["path,participant_id,sequence_id,sign"]
    number = 0
    for s, sign in enumerate(signs):
        shape = np.random.default_rng(100 + s).uniform(0.3, 0.7, (21, 3))
        pose = np.random.default_rng(200).uniform(0.2, 0.8, (33, 3))
        for signer in signers:
            for _ in range(per_signer):
                number += 1
                rng = np.random.default_rng(number)
                drift = np.linspace(0, 0.1 * (s + 1), frames)[:, None, None]
                path = f"train_landmark_files/{signer}/{number}.parquet"
                write_sequence(folder / path, {
                    "right_hand": shape + drift + rng.normal(0, 0.004, (frames, 21, 3)),
                    "pose": pose + rng.normal(0, 0.004, (frames, 33, 3)),
                })
                lines.append(f"{path},{signer},{number},{sign}")
    (folder / "train.csv").write_text("\n".join(lines) + "\n")


def test_a_sequence_becomes_a_raw_clip_trimmed_to_where_a_hand_is_seen(tmp_path):
    rng = np.random.default_rng(0)
    right = rng.uniform(0.3, 0.7, (6, 21, 3))
    right[[0, 5]] = np.nan  # no hand in the first and last frame
    right[2, 4, 1] = np.nan  # one bad coordinate: the whole hand is left out of that frame
    pose = rng.uniform(0.2, 0.8, (6, 33, 3))
    face = rng.uniform(0.2, 0.8, (6, 468, 3))
    write_sequence(tmp_path / "a.parquet", {"right_hand": right, "pose": pose, "face": face})

    clip = sequence_to_clip(tmp_path / "a.parquet")
    assert clip.shape == (4, FRAME_SIZE)
    assert np.allclose(clip[0, RIGHT_HAND:POSE], right[1].ravel())
    assert not clip[1, RIGHT_HAND:POSE].any() and clip[2, RIGHT_HAND:POSE].any()
    assert np.allclose(clip[0, POSE:FACE], pose[1, :25].ravel())  # the app keeps the first 25 pose points
    assert not clip[:, LEFT_HAND:RIGHT_HAND].any() and not clip[:, FACE:].any()


def test_long_sequences_are_thinned_and_unusable_files_are_skipped(tmp_path):
    hand = np.random.default_rng(0).uniform(0.3, 0.7, (30, 21, 3))
    write_sequence(tmp_path / "long.parquet", {"left_hand": hand}, index_name="landmark_idx")
    clip = sequence_to_clip(tmp_path / "long.parquet", max_frames=8)
    assert clip.shape == (8, FRAME_SIZE)
    assert np.allclose(clip[0, LEFT_HAND:RIGHT_HAND], hand[0].ravel())
    assert np.allclose(clip[-1, LEFT_HAND:RIGHT_HAND], hand[-1].ravel())

    write_sequence(tmp_path / "nohand.parquet", {"pose": np.full((3, 33, 3), 0.5)})
    assert sequence_to_clip(tmp_path / "nohand.parquet") is None
    (tmp_path / "broken.parquet").write_text("not parquet")
    assert sequence_to_clip(tmp_path / "broken.parquet") is None
    assert sequence_to_clip(tmp_path / "missing.parquet") is None
    pq.write_table(pa.table({"frame": [1], "other": [2]}), tmp_path / "columns.parquet")
    assert sequence_to_clip(tmp_path / "columns.parquet") is None


def test_clips_are_labelled_apart_from_taught_signs_and_can_be_limited_per_sign(tmp_path):
    write_dataset(tmp_path)
    with open(tmp_path / "train.csv", "a") as file:
        file.write("train_landmark_files/11/gone.parquet,11,999,hello\n")

    clips = load_sequences(tmp_path)
    assert len(clips) == 27
    assert {c.label for c in clips} == {LABEL_PREFIX + s for s in ("hello", "dog", "TV")}
    assert {c.signer for c in clips} == {SIGNER_PREFIX + s for s in ("11", "22", "33")}

    limited = load_sequences(tmp_path, per_sign=2)
    assert sorted(c.label for c in limited) == sorted([LABEL_PREFIX + s for s in ("hello", "dog", "TV")] * 2)


def test_pretrain_on_both_datasets_holds_out_a_whole_signer(tmp_path):
    write_samples(tmp_path / "letters")
    write_dataset(tmp_path / "signs")
    report = pretrain([tmp_path / "letters", tmp_path / "signs"], tmp_path / "p.pt", epochs=2, seed=0)

    assert report["source"] == f"{public_data.SOURCE} + {SOURCE}"
    assert report["labels"] == sorted(["A", "B", "C", IDLE_LABEL, *(LABEL_PREFIX + s for s in ("hello", "dog", "TV"))])
    assert report["samples"] == 24 + 27
    # One of three signers (nine clips) and one of eight samples of each letter.
    assert report["test_samples"] == 9 + 3
    assert "1 of 3 signers held out" in report["held_out"] and 0 <= report["top1"] <= 1
