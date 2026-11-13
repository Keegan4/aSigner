"""Checks the pretraining path end to end on made-up samples. Says nothing about real accuracy."""
import json

import numpy as np
import pytest
import torch

from checkpoint import load_into, save_checkpoint
from pretrain import pretrain
from public_data import FRAMES_PER_CLIP, SOURCE, load_samples
from test_train import write_clips
from train import SignNet, train

from app.recognition.features import FRAME_SIZE, IDLE_LABEL, LEFT_HAND, POSE, RIGHT_HAND
from app.recognition.onnx_recogniser import load_recogniser


def write_samples(folder, letters="ABC", per_letter=8):
    rng = np.random.default_rng(0)
    for index, letter in enumerate(letters):
        (folder / letter).mkdir(parents=True)
        shape = np.random.default_rng(100 + index).uniform(0.4, 0.6, (21, 3))
        for n in range(per_letter):
            points = shape + rng.normal(0, 0.004, (21, 3))
            (folder / letter / f"{n}.json").write_text(
                json.dumps([{"x": x, "y": y, "z": z} for x, y, z in points.tolist()])
            )


def test_samples_become_clips_of_one_still_hand_and_bad_files_are_skipped(tmp_path):
    write_samples(tmp_path, per_letter=2)
    (tmp_path / "A" / "short.json").write_text(json.dumps([{"x": 0, "y": 0, "z": 0}] * 20))
    (tmp_path / "A" / "nan.json").write_text(json.dumps([{"x": float("nan"), "y": 0, "z": 0}] * 21))
    (tmp_path / "A" / "broken.json").write_text("{not json")
    (tmp_path / "A" / "wrong.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "0.json").write_text((tmp_path / "A" / "0.json").read_text())

    clips = load_samples(tmp_path)
    assert [c.label for c in clips] == ["A", "A", "B", "B", "C", "C"]
    frames = clips[0].frames
    assert frames.shape == (FRAMES_PER_CLIP, FRAME_SIZE)
    assert frames[:, RIGHT_HAND:POSE].any() and (frames[0] == frames[-1]).all()
    assert not frames[:, LEFT_HAND:RIGHT_HAND].any() and not frames[:, POSE:].any()


def test_pretrain_then_train_from_the_checkpoint(tmp_path):
    write_samples(tmp_path / "public")
    checkpoint = tmp_path / "pretrained.pt"
    report = pretrain(tmp_path / "public", checkpoint, epochs=2, seed=0)
    assert report["labels"] == ["A", "B", "C", IDLE_LABEL] and report["source"] == SOURCE

    write_clips(tmp_path / "clips")
    out = tmp_path / "models" / "signs.onnx"
    trained = train(tmp_path / "clips", out, epochs=2, seed=0, test_signer=None,
                    runs=tmp_path / "runs", init=checkpoint)
    assert trained["info"]["pretrained"] == SOURCE
    assert load_recogniser(str(out)).info["pretrained"] == SOURCE

    # Without a checkpoint nothing about the model says it was pretrained.
    plain = train(tmp_path / "clips", out, epochs=1, seed=0, test_signer=None, runs=tmp_path / "runs")
    assert "pretrained" not in plain["info"]


def test_checkpoint_rows_follow_labels_not_positions(tmp_path):
    torch.manual_seed(0)
    source = SignNet(3)
    save_checkpoint(source, ["A", "B", IDLE_LABEL], "somewhere", tmp_path / "c.pt")

    torch.manual_seed(1)
    target = SignNet(4)
    fresh = target.head[-1].weight.detach().clone()
    assert load_into(target, ["B", "NEW", IDLE_LABEL, "A"], tmp_path / "c.pt") == "somewhere"

    old, new = source.head[-1], target.head[-1]
    assert torch.equal(new.weight[0], old.weight[1]) and torch.equal(new.bias[0], old.bias[1])  # B
    assert torch.equal(new.weight[3], old.weight[0])  # A
    assert torch.equal(new.weight[2], old.weight[2])  # idle
    assert torch.equal(new.weight[1], fresh[1])  # a label the checkpoint never saw
    assert torch.equal(target.regions[1].encoder[0].weight, source.regions[1].encoder[0].weight)


def test_a_checkpoint_for_another_network_is_refused(tmp_path):
    save_checkpoint(SignNet(3, width=160), ["A", "B", IDLE_LABEL], "somewhere", tmp_path / "small.pt")
    with pytest.raises(SystemExit, match="does not fit this model"):
        load_into(SignNet(3), ["A", "B", IDLE_LABEL], tmp_path / "small.pt")


def write_casl(folder, signs=("G001", "G002"), per_sign=4, frames=64):
    """Made-up CASL-W60 parquet files: (frames, 225) pose + left hand + right hand per clip."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(1)
    (folder / "data").mkdir(parents=True)
    for split in ("train", "test"):
        rows, labels = [], []
        for index, sign in enumerate(signs):
            for _ in range(per_sign):
                pose = rng.normal(0, 0.2, (frames, 33, 3))
                left = np.zeros((frames, 21, 3)) if index == 0 else rng.normal(0, 0.1, (frames, 21, 3))
                right = rng.normal(0.1 * index, 0.1, (frames, 21, 3))
                rows.append(np.concatenate([pose.reshape(frames, -1), left.reshape(frames, -1),
                                            right.reshape(frames, -1)], axis=1).astype(np.float32).tolist())
                labels.append(sign)
        pq.write_table(pa.table({"landmarks": rows, "label": labels}),
                       folder / "data" / f"{split}-00000-of-00001.parquet")


def test_casl_clips_keep_their_split_pose_and_hands(tmp_path):
    from public_data import CASL_PREFIX, load_casl

    write_casl(tmp_path / "casl")
    train_clips, test_clips = load_casl(tmp_path / "casl")
    assert len(train_clips) == len(test_clips) == 8
    assert {c.label for c in train_clips} == {CASL_PREFIX + "G001", CASL_PREFIX + "G002"}
    assert {c.signer for c in test_clips} == {"casl-test"}

    import pyarrow.parquet as pq

    first = np.asarray(pq.read_table(tmp_path / "casl" / "data" / "train-00000-of-00001.parquet")
                       .column("landmarks")[0].as_py(), dtype=np.float32)
    frames = train_clips[0].frames
    assert frames.shape == (64, FRAME_SIZE)
    assert np.array_equal(frames[:, POSE:POSE + 75], first[:, :75])  # pose points 0-24
    assert np.array_equal(frames[:, RIGHT_HAND:POSE], first[:, 162:])
    assert not frames[:, LEFT_HAND:RIGHT_HAND].any()  # this sign's left hand was never seen
    assert not frames[:, POSE + 75:].any()  # no face
    assert load_casl(tmp_path / "missing") == ([], [])


def test_pretrain_with_casl_records_both_sources_and_every_epoch(tmp_path):
    from public_data import CASL_PREFIX, CASL_SOURCE

    write_samples(tmp_path / "public")
    write_casl(tmp_path / "casl")
    report = pretrain(tmp_path / "public", tmp_path / "pretrained.pt", epochs=2, seed=0, casl=tmp_path / "casl")
    assert report["source"] == f"{SOURCE} + {CASL_SOURCE}"
    assert CASL_PREFIX + "G001" in report["labels"] and "A" in report["labels"]
    assert [row["epoch"] for row in report["history"]] == [1, 2]
    assert set(report["history"][0]) == {"epoch", "train", "asl-now", "casl-w60"}
    assert 0 <= report["history"][-1]["casl-w60"]["f1"] <= 1
    assert json.loads((tmp_path / "pretrained.json").read_text())["history"] == report["history"]


def test_wlasl_points_map_to_pose_and_hands_and_each_word_keeps_a_fixed_test_clip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from public_data import WLASL_PREFIX, load_wlasl, wlasl_to_clip

    rng = np.random.default_rng(2)
    frames = rng.uniform(0.2, 0.8, (10, 76, 3)).astype(np.float32)
    frames[:4, 34:55] = 0  # left hand not tracked at first
    clip = wlasl_to_clip(frames).reshape(10, -1)
    pose = clip[:, POSE:POSE + 75].reshape(10, 25, 3)
    assert np.array_equal(pose[:, [0, 11, 12, 13, 14, 15, 16, 23, 24]], frames[:, 25:34])
    assert np.array_equal(pose[:, 5], frames[:, 25])  # an eye, at the nose
    assert np.array_equal(pose[:4, 19], pose[:4, 15])  # left index at the wrist while the hand is missing
    assert np.array_equal(pose[4:, 19], frames[4:, 34 + 5])  # then at the hand's index knuckle
    assert np.array_equal(clip[:, RIGHT_HAND:POSE], frames[:, 55:].reshape(10, -1))
    assert not clip[:4, LEFT_HAND:RIGHT_HAND].any() and not clip[:, POSE + 75:].any()

    (tmp_path / "data").mkdir()
    rows = [rng.uniform(0.2, 0.8, (8, 76, 3)).tolist() for _ in range(9)]
    labels = ["book"] * 6 + ["drink"] * 2 + ["unknown"]
    pq.write_table(pa.table({"landmarks": rows, "label": labels}), tmp_path / "data" / "train-0.parquet")
    train_clips, test_clips = load_wlasl(tmp_path)
    assert [c.label for c in test_clips] == [WLASL_PREFIX + "book"]  # only words with five or more clips
    assert sorted({c.label for c in train_clips}) == [WLASL_PREFIX + "book", WLASL_PREFIX + "drink"]
    assert len(train_clips) == 7
    assert [c.frames.tobytes() for c in load_wlasl(tmp_path)[1]] == [c.frames.tobytes() for c in test_clips]
