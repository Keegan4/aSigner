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


def test_pretrain_then_train_from_the_checkpoint(tmp_path, monkeypatch):
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

    # Freezing the backbone leaves every weight outside the head exactly as pretrained.
    import train as train_module
    built = []
    real = train_module.SignNet
    monkeypatch.setattr(train_module, "SignNet", lambda *a, **k: built.append(real(*a, **k)) or built[-1])
    train(tmp_path / "clips", out, epochs=2, seed=0, test_signer=None, runs=tmp_path / "runs",
          init=checkpoint, freeze_backbone=True)
    before = real(len(built[0].head[-1].weight), width=320)
    load_into(before, ["A", "B", "C", IDLE_LABEL], checkpoint)
    after = built[0]
    for name, tensor in after.state_dict().items():
        if not name.startswith("head."):
            assert torch.equal(tensor.cpu(), before.state_dict()[name]), name
    assert not torch.equal(after.head[0].weight.cpu(), before.head[0].weight)
    with pytest.raises(SystemExit):
        train(tmp_path / "clips", out, epochs=1, seed=0, test_signer=None, runs=tmp_path / "runs",
              freeze_backbone=True)

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
