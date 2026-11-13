"""Checks the pipeline end to end on made-up clips. Says nothing about real accuracy."""
import numpy as np
import pytest
import torch

from data import Clip, augment, mirror, to_model_input, transition
from train import RegionEncoder, SignNet, export, pick_device, probabilities, threshold_curve, train

from app.recognition.features import (
    CLIP_FRAMES, COORDINATE_SIZE, FRAME_SIZE, LEFT_HAND, NORMALISED_PARTS,
    NORMALISED_SIZE, POSE, PRESENCE, RIGHT_HAND,
)
from app.recognition.onnx_recogniser import load_recogniser


def fake_clip(rng, label_index: int, frames: int = 50) -> np.ndarray:
    """A right hand whose shape depends on the label, with a body, jittered per clip."""
    shape = np.random.default_rng(100 + label_index).uniform(0.4, 0.6, 63)
    clip = np.zeros((frames, FRAME_SIZE), dtype=np.float32)
    clip[:, RIGHT_HAND:POSE] = shape + rng.normal(0, 0.004, (frames, 63))
    clip[:, POSE:POSE + 75] = np.random.default_rng(7).uniform(0.3, 0.7, 75)
    return clip


def write_clips(folder, signers=("u1", "u2"), per_sign=12, counts=None):
    rng = np.random.default_rng(0)
    for index, label in enumerate("ABC"):
        (folder / label).mkdir(parents=True)
        for signer in signers:
            for n in range((counts or {}).get(label, per_sign)):
                np.savez_compressed(folder / label / f"{signer}_{n}.npz", frames=fake_clip(rng, index),
                                    label=label, signer=signer, fps=np.float32(25))


def test_mirror_swaps_hands_and_is_its_own_inverse():
    clip = fake_clip(np.random.default_rng(1), 0)
    flipped = mirror(clip)
    assert not flipped[:, RIGHT_HAND:POSE].any()  # the right hand became the left
    assert flipped[:, LEFT_HAND:RIGHT_HAND].any()
    assert np.allclose(mirror(flipped), clip, atol=1e-6)


def test_augment_keeps_missing_parts_missing():
    rng = np.random.default_rng(2)
    clip = fake_clip(rng, 1)
    for _ in range(20):
        out = augment(clip, rng)
        assert out.shape[1] == FRAME_SIZE and len(out) >= 2
        # Only one hand was recorded, so exactly one hand may be present afterwards.
        assert out[:, LEFT_HAND:RIGHT_HAND].any() != out[:, RIGHT_HAND:POSE].any()
        assert to_model_input(out).shape == (CLIP_FRAMES, NORMALISED_SIZE)


def test_signnet_backpropagates_through_cls_positions_and_every_frame():
    torch.manual_seed(0)
    model = SignNet(classes=4).eval()
    clips = torch.randn(4, CLIP_FRAMES, NORMALISED_SIZE)
    clips[..., PRESENCE] = 1
    clips.requires_grad_()
    logits = model(clips)
    assert logits.shape == (4, 4)
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()

    for gradient in (model.cls_token.grad, model.positions.grad, clips.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    # Classification can depend on distant frames, including the first and last.
    assert (clips.grad.abs().sum(dim=2) > 0).all()
    for region in model.regions:
        assert region.encoder[0].weight.grad.abs().sum() > 0


def test_signnet_distinguishes_frame_order_using_positions():
    torch.manual_seed(1)
    model = SignNet(classes=4).eval()
    clip = torch.randn(1, CLIP_FRAMES, NORMALISED_SIZE)
    clip[..., PRESENCE] = 1
    with torch.no_grad():
        original = model(clip)
        assert not torch.allclose(original, model(clip.flip(1)), atol=1e-6)
        model.positions.zero_()
        assert not torch.allclose(original, model(clip), atol=1e-6)


def test_motion_ignores_tracking_loss_recovery_and_first_frame():
    region = RegionEncoder((4,), 80)
    clip = torch.zeros(1, 5, NORMALISED_SIZE)
    clip[0, :, NORMALISED_PARTS[4]] = torch.tensor([1., 3., 100., 9., 10.])[:, None]
    clip[0, :, COORDINATE_SIZE + 4] = torch.tensor([1., 1., 0., 1., 1.])
    features, flags = region.inputs(clip)
    torch.testing.assert_close(features[0, :, 0], torch.tensor([1., 3., 0., 9., 10.]))
    expected = torch.tensor([0., 2., 0., 0., 1.])[:, None].expand(-1, 75)
    torch.testing.assert_close(features[0, :, 75:150], expected)
    torch.testing.assert_close(features[0, :, -1], torch.tensor([0., 1., 0., 0., 1.]))
    stationary = clip[:, :1].expand(-1, 5, -1)
    assert not region.inputs(stationary)[0][..., 75:150].any()


def test_hand_encoder_groups_shape_and_placement_without_the_other_hand():
    clip = torch.zeros(1, 2, NORMALISED_SIZE)
    clip[..., NORMALISED_PARTS[0]] = 2
    clip[..., NORMALISED_PARTS[1]] = 100
    clip[..., NORMALISED_PARTS[2]] = 3
    clip[..., NORMALISED_PARTS[3]] = 200
    clip[..., PRESENCE] = 1
    features, flags = RegionEncoder((0, 2), 80).inputs(clip)
    assert features.shape == (1, 2, 256)
    assert (features[..., :63] == 2).all()
    assert (features[..., 63:126] == 3).all()
    assert not features[..., 126:252].any()  # stationary, including the first frame
    assert (flags == 1).all()


def test_region_encoders_have_two_layers():
    for parts, inputs in (((0, 2), 256), ((1, 3), 256), ((4,), 152), ((5,), 112)):
        linears = [m for m in RegionEncoder(parts, 80).encoder if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == [(inputs, 80), (80, 80)]


def test_parameter_count_matches_the_readme():
    for classes in (2, 101):
        model = SignNet(classes)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 2_005_920 + 161 * classes


def test_missing_regions_are_finite_and_ignore_invalid_coordinates():
    torch.manual_seed(3)
    model = SignNet(classes=4).eval()
    clip = torch.randn(1, CLIP_FRAMES, NORMALISED_SIZE)
    clip[..., PRESENCE] = 0
    logits = model(clip)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, model(torch.zeros_like(clip)))
    logits.sum().backward()
    for region in model.regions:
        assert region.missing.grad is not None and region.missing.grad.abs().sum() > 0


def test_attention_export_matches_pytorch_on_different_clips(tmp_path):
    torch.manual_seed(2)
    model = SignNet(classes=4).eval()
    # Non-default saved statistics exercise BatchNorm export without training.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.running_mean.fill_(0.2)
                layer.running_var.fill_(0.8)
    rng = np.random.default_rng(2)
    probe = rng.normal(size=(CLIP_FRAMES, NORMALISED_SIZE)).astype(np.float32)
    probe[:, PRESENCE] = 1
    partial = probe.copy()
    partial[:, COORDINATE_SIZE] = 0
    partial[10:20, COORDINATE_SIZE + 1:] = 0
    lost = probe.copy()
    lost[10:20] = 0
    out = tmp_path / "signs.onnx"
    export(model, out, ["A", "B", "C", "_idle"], None, probe)
    recogniser = load_recogniser(str(out))
    for clip in (probe, probe[::-1].copy(), np.zeros_like(probe), partial, lost):
        with torch.no_grad():
            expected = torch.softmax(model(torch.from_numpy(clip)[None]), dim=1)[0].numpy()
        np.testing.assert_allclose(recogniser.predict(clip), expected, atol=1e-4)


def test_train_exports_a_model_the_server_can_load(tmp_path):
    write_clips(tmp_path / "clips")
    out = tmp_path / "models" / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=30, seed=0, test_signer="u2", runs=tmp_path / "runs")

    assert report["labels"] == ["A", "B", "C", "_idle"]  # no-sign is always a class
    assert "u2 held out" in report["split"]
    assert report["signs_without_test_clips"] == []
    assert report["top1"] > 0.9  # the made-up signs are easy to tell apart
    assert len(report["threshold_curve"]) == 14
    assert 0.5 <= report["pass_score"] <= 0.95  # 36 held-out clips is enough to choose one
    chosen = next(row for row in report["threshold_curve"] if row["pass_score"] == report["pass_score"])
    assert chosen["false_accept"] <= 0.05 or report["pass_score"] == 0.95

    # One file, nothing left half-written beside it.
    assert [p.name for p in out.parent.iterdir()] == ["signs.onnx"]
    recogniser = load_recogniser(str(out))
    assert recogniser.labels == ["A", "B", "C", "_idle"]
    assert recogniser.pass_score == report["pass_score"]
    assert recogniser.info == report["info"]
    assert recogniser.info["clips"] == 36 and recogniser.info["signers"] == 1  # u1 only; u2 was held out
    assert recogniser.info["test_clips"] == 36 and recogniser.info["untested_signs"] == 0
    assert recogniser.info["top1"] == report["top1"]
    probabilities = recogniser.predict(to_model_input(fake_clip(np.random.default_rng(3), 2)))
    assert recogniser.labels[int(probabilities.argmax())] == "C"
    assert list((tmp_path / "runs").glob("*/confusion.csv"))


@pytest.mark.parametrize("cuda, xpu, mps, expected", [
    (True, True, True, "cuda"), (False, True, True, "xpu"), (False, False, True, "mps"), (False, False, False, "cpu"),
])
def test_device_is_a_gpu_whenever_pytorch_sees_one(monkeypatch, cuda, xpu, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert pick_device() == torch.device(expected)


@pytest.mark.skipif(pick_device().type == "cpu", reason="needs a GPU PyTorch can use")
def test_training_on_the_gpu_exports_a_cpu_model_the_server_can_load(tmp_path):
    write_clips(tmp_path / "clips", per_sign=6)
    out = tmp_path / "signs.onnx"
    train(tmp_path / "clips", out, epochs=2, seed=0, test_signer="u2", runs=tmp_path / "runs")
    assert load_recogniser(str(out)).labels == ["A", "B", "C", "_idle"]


def test_probabilities_come_back_on_the_cpu_from_any_device():
    torch.manual_seed(0)
    model = SignNet(classes=3).to(pick_device())
    rng = np.random.default_rng(0)
    probs = probabilities(model, [Clip(fake_clip(rng, 0), "A", "u1"), Clip(fake_clip(rng, 1), "B", "u1")])
    assert isinstance(probs, np.ndarray) and probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_training_does_not_leave_a_singleton_batch(tmp_path):
    # A conventional split into batches of 64 would leave BatchNorm one clip.
    write_clips(tmp_path / "clips", counts={"A": 21, "B": 22, "C": 22})
    out = tmp_path / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=1, seed=0, test_signer="u2", runs=tmp_path / "runs")
    assert report["clips"] == 130 and report["test_clips"] == 65
    assert load_recogniser(str(out)).labels == report["labels"]


def test_training_explains_when_batchnorm_has_only_one_training_clip(tmp_path):
    folder = tmp_path / "clips"
    write_clips(folder, per_sign=1)
    (folder / "B" / "u1_0.npz").unlink()
    (folder / "C" / "u1_0.npz").unlink()
    with pytest.raises(SystemExit, match="at least two training clips for batch normalisation"):
        train(folder, tmp_path / "signs.onnx", epochs=1, seed=0,
              test_signer="u2", runs=tmp_path / "runs")


def test_transition_puts_the_sign_last():
    rng = np.random.default_rng(4)
    sign, before = fake_clip(rng, 0), fake_clip(rng, 1)
    for _ in range(50):
        frames, fraction = transition(sign, before, rng)
        shown = round(fraction * len(frames))
        assert 0.3 <= fraction <= 1.0
        assert np.array_equal(frames[len(frames) - shown :], sign[:shown])
        assert np.array_equal(frames[: len(frames) - shown], before[len(before) - (len(frames) - shown) :])


def test_threshold_curve_counts_rejects_and_accepts():
    labels = ["A", "B", "_idle"]
    probs = np.array([
        [0.9, 0.05, 0.05],  # A, clearly right
        [0.5, 0.4, 0.1],  # A, right but unsure: rejected above 0.5
        [0.2, 0.7, 0.1],  # A signed, looks like B: would pass as B up to 0.7
        [0.1, 0.1, 0.8],  # idle: never a target, never an accept
    ])
    curve = {row["pass_score"]: row for row in threshold_curve(probs, np.array([0, 0, 0, 2]), labels)}
    assert curve[0.3]["false_reject"] == 1 / 3 and curve[0.3]["false_accept"] == 1 / 4
    assert curve[0.6]["false_reject"] == 2 / 3 and curve[0.6]["false_accept"] == 1 / 4
    assert curve[0.75]["false_accept"] == 0


def test_report_names_signs_that_were_never_tested(tmp_path, capsys):
    """One signer, one clip for B and C: accuracy rests on A alone and must say so."""
    write_clips(tmp_path / "clips", signers=("u1",), counts={"A": 2, "B": 1, "C": 1})
    report = train(tmp_path / "clips", tmp_path / "m" / "signs.onnx", epochs=2, seed=0,
                   test_signer=None, runs=tmp_path / "runs")
    assert report["test_clips"] == 1
    assert report["signs_without_test_clips"] == ["B", "C"]
    assert report["pass_score"] is None  # far too few clips to choose one
    assert "2 of 3 signs had no held-out clip" in capsys.readouterr().out


def test_train_runs_from_a_windows_console(tmp_path):
    """train.py once died at export on Windows: the exporter prints a character
    cp1252 cannot encode. Pytest's own output capture hides that, so run the
    script as a real process with that encoding forced."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    write_clips(tmp_path / "clips", per_sign=4)
    out = tmp_path / "m" / "signs.onnx"
    done = subprocess.run(
        [sys.executable, "train.py", "--clips", str(tmp_path / "clips"), "--out", str(out),
         "--epochs", "1", "--runs", str(tmp_path / "runs")],
        cwd=Path(__file__).resolve().parent, capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert done.returncode == 0, done.stderr.decode("cp1252", "replace")[-2000:]
    assert out.is_file()


def test_info_records_where_clips_came_from(tmp_path):
    """A model trained on converted outside footage must say so in the file itself."""
    write_clips(tmp_path / "clips", signers=("u1", "signbank"), per_sign=3)
    out = tmp_path / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=1, seed=0, test_signer="u1", runs=tmp_path / "runs")
    assert report["info"]["sources"] == {"signbank": 9}
    assert load_recogniser(str(out)).info["sources"] == {"signbank": 9}

    both = tmp_path / "both.onnx"
    report = train(tmp_path / "clips", both, epochs=1, seed=0, test_signer="x", runs=tmp_path / "runs")
    assert report["info"]["sources"] == {"recorded": 9, "signbank": 9}
