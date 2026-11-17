"""Checks the pipeline end to end on made-up clips. Says nothing about real accuracy."""
import numpy as np
import pytest
import torch

from data import (
    Clip, augment, close_pool, live_windows, load_clips, middle_window, mirror, to_model_input, window,
    window_frames,
)
from train import (
    HAND_BONES, HandGraph, RegionEncoder, SignNet, export, hand_adjacency, pick_device, probabilities,
    threshold_curve, train, training_batch,
)

from app.recognition.features import (
    CLIP_FRAMES, COORDINATE_SIZE, FRAME_SIZE, IDLE_LABEL, LEFT_HAND, NORMALISED_PARTS,
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


def test_region_encoders_halve_their_input_then_give_hands_the_most_room():
    expected = [(256, 128, 120), (256, 128, 120), (152, 76, 40), (112, 56, 40)]
    for region, (inputs, hidden, width) in zip(SignNet(classes=4).regions, expected):
        linears = [m for m in region.encoder if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == [(inputs, hidden), (hidden, width)]
        assert region.missing.shape == (width,)


def test_parameter_count_matches_the_readme():
    for classes in (2, 101):
        model = SignNet(classes)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 2_037_668 + 161 * classes


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
    assert report["window_top1"] > 0.9 and report["windows"] == 36 * 2  # and so is every one-second window of them
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
    signs = recogniser.info["signs"]
    assert list(signs) == ["A", "B", "C", "_idle"]
    assert sum(s["clips"] for s in signs.values()) == 36
    assert sum(s["test_clips"] for s in signs.values()) == 36
    tested = [s for s in signs.values() if s["test_clips"]]
    assert all(0 <= s["top1"] <= 1 for s in tested)
    assert all(s["top1"] is None for s in signs.values() if not s["test_clips"])
    # The per-sign accuracies weighted by held-out clips add up to the overall one.
    assert abs(sum(s["top1"] * s["test_clips"] for s in tested) / 36 - report["top1"]) < 1e-9
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


def numbered_clip(value: int, label: str, frames: int = 40) -> Clip:
    """Ten frames a second; every frame holds `value` and then its own index."""
    raw = np.full((frames, FRAME_SIZE), value, dtype=np.float32)
    raw[:, 1] = np.arange(frames)
    return Clip(raw, label, "u1", fps=10.0)


def test_a_window_is_a_stretch_from_anywhere_in_the_clip():
    rng = np.random.default_rng(4)
    clip = numbered_clip(1, "A")
    starts = set()
    for _ in range(200):
        frames, label = window(clip, None, rng)
        assert label == "A"
        assert len(frames) == 11  # one second at ten frames a second, both ends included
        start = int(frames[0, 1])
        assert np.array_equal(frames, clip.frames[start : start + len(frames)])
        starts.add(start)
    assert 0 in starts and max(starts) >= 29  # the start of the sign and its end, not only the start
    short = Clip(clip.frames[:2], "A", "u1", fps=10.0)
    assert np.array_equal(window(short, None, rng)[0], short.frames)


def test_a_straddling_window_is_labelled_by_what_fills_it():
    rng = np.random.default_rng(4)
    a, b = numbered_clip(1, "A"), numbered_clip(2, "B")
    seen = set()
    for _ in range(300):
        frames, label = window(a, b, rng)
        own = frames[:, 0] == 1
        assert 0 < own.sum() < len(frames) and np.count_nonzero(np.diff(own)) == 1  # both clips, one join
        share = own.mean()
        assert label == ("A" if share >= 0.6 else "B" if 1 - share >= 0.6 else IDLE_LABEL)
        if own[0]:  # A ends, B begins: the end of A meets the start of B
            assert frames[own][-1, 1] == 39 and frames[~own][0, 1] == 0
        else:
            assert frames[~own][-1, 1] == 39 and frames[own][0, 1] == 0
        seen.add((label, bool(own[0])))
    assert seen == {(label, first) for label in ("A", "B", IDLE_LABEL) for first in (True, False)}
    assert window(a, numbered_clip(2, "A"), rng)[1] == "A"  # the same sign twice is still that sign


def test_every_window_is_one_second_at_the_rate_the_clip_was_tracked_at(tmp_path):
    write_clips(tmp_path, signers=("u1",), per_sign=1)
    clip = load_clips(tmp_path)[0]  # 50 frames at 25 a second
    assert clip.fps == 25 and window_frames(clip) == 26  # one second, both ends included
    assert window_frames(numbered_clip(1, "A")) == 11  # what the translator holds at ten frames a second
    assert window_frames(Clip(clip.frames, "A", "u1")) == 26  # no rate given: taken to last two seconds
    windows = live_windows(clip)
    assert len(windows) == 2 and all(len(w) == 26 for w in windows)
    assert np.array_equal(windows[1], clip.frames[13:39])
    assert np.array_equal(middle_window(clip), clip.frames[12:38])  # what a held-out clip is scored on
    # A one-second take is used as it is.
    take = Clip(clip.frames[:25], "A", "u1", fps=25.0)
    assert len(live_windows(take)) == 1 and len(live_windows(take)[0]) == 25
    assert np.array_equal(middle_window(take), take.frames)
    assert np.array_equal(window(take, None, np.random.default_rng(0))[0], take.frames)


def test_augmenting_on_several_cores_gives_the_same_batch_as_on_one():
    rng = np.random.default_rng(5)
    clips = [Clip(fake_clip(rng, i % 3), "ABC"[i % 3], "u1") for i in range(24)]
    labels = ["A", "B", "C", IDLE_LABEL]
    try:
        x_one, y_one = training_batch(clips, labels, np.random.default_rng(9), workers=1)
        x_many, y_many = training_batch(clips, labels, np.random.default_rng(9), workers=3)
        again, _ = training_batch(clips, labels, np.random.default_rng(10), workers=3)
    finally:
        close_pool()
    assert x_one.shape == (24, CLIP_FRAMES, NORMALISED_SIZE)
    assert torch.equal(x_one, x_many) and torch.equal(y_one, y_many)
    assert not torch.equal(x_many, again)  # a new epoch is augmented afresh
    # Some windows straddle two signs and take the other sign's label.
    assert any(labels[i] != clip.label for i, clip in zip(y_one, clips))


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


def test_signnet_can_leave_out_the_temporal_conv():
    torch.manual_seed(4)
    with_conv, without = SignNet(classes=4), SignNet(classes=4, temporal_conv=False)
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    # LayerNorm 640, depthwise 3·320 + 320, pointwise 320·320 + 320.
    assert count(with_conv) - count(without) == 104_640
    assert not any(name.startswith("temporal") for name, _ in without.named_parameters())
    logits = without.eval()(torch.randn(2, CLIP_FRAMES, NORMALISED_SIZE))
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()


def test_a_model_without_the_temporal_conv_exports_and_says_so(tmp_path):
    write_clips(tmp_path / "clips", per_sign=4)
    out = tmp_path / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=2, seed=0, test_signer="u2", runs=tmp_path / "runs",
                   temporal_conv=False)
    assert report["temporal_conv"] is False
    assert load_recogniser(str(out)).info["temporal_conv"] is False


def test_ablation_pairs_each_seed_and_leaves_no_model_behind(tmp_path):
    from ablate import ablate

    write_clips(tmp_path / "clips", per_sign=4)
    result = ablate(tmp_path / "clips", seeds=[0, 1], epochs=2, test_signer="u2", runs=tmp_path / "runs")
    assert set(result["variants"]) == {"with temporal conv", "without temporal conv"}
    for summary in result["variants"].values():
        assert len(summary["top1"]["runs"]) == 2 and len(summary["top5"]["runs"]) == 2
    paired = [a - b for a, b in zip(result["variants"]["with temporal conv"]["top1"]["runs"],
                                    result["variants"]["without temporal conv"]["top1"]["runs"])]
    assert result["difference"]["top1"]["runs"] == pytest.approx(paired)
    assert list((tmp_path / "runs").glob("ablation-*/ablation.json"))
    assert not list(tmp_path.rglob("*.onnx"))  # the served model is never touched


def test_hand_adjacency_links_each_landmark_to_itself_and_its_bones():
    adjacency = hand_adjacency()
    links = adjacency > 0
    assert adjacency.shape == (21, 21) and torch.equal(adjacency, adjacency.T)
    assert len(set(HAND_BONES)) == 21 and int(links.sum()) == 21 + 2 * 21
    assert links.diagonal().all() and all(links[a, b] for a, b in HAND_BONES)
    # The wrist has three bones (thumb, index and little finger) plus itself; a fingertip has one.
    assert float(adjacency[0, 0]) == pytest.approx(1 / 4) and float(adjacency[8, 8]) == pytest.approx(1 / 2)
    assert float(adjacency[7, 8]) == pytest.approx(1 / (3 * 2) ** 0.5)  # a mid-finger joint has two bones


def test_one_graph_convolution_reaches_only_the_next_landmark():
    torch.manual_seed(5)
    graph = HandGraph(width=120, joint_features=12).eval()
    joints = torch.randn(1, 1, 21, 12)
    moved = joints.clone()
    moved[..., 8, :] += 1.0  # the index fingertip
    with torch.no_grad():
        once = (graph.mix(moved, graph.first) - graph.mix(joints, graph.first)).abs().sum(dim=-1)[0, 0]
        twice = (graph.mix(graph.mix(moved, graph.first), graph.second)
                 - graph.mix(graph.mix(joints, graph.first), graph.second)).abs().sum(dim=-1)[0, 0]
    assert set(once.nonzero().flatten().tolist()) == {7, 8}
    assert set(twice.nonzero().flatten().tolist()) == {6, 7, 8}


def test_hand_inputs_are_regrouped_by_landmark():
    encoder = RegionEncoder((1, 3), width=120, graph=True)
    clip = torch.zeros(1, 2, NORMALISED_SIZE)
    clip[..., [COORDINATE_SIZE + 1, COORDINATE_SIZE + 3]] = 1.0
    shape, placement = NORMALISED_PARTS[1], NORMALISED_PARTS[3]
    clip[0, :, shape] = torch.arange(63.0)
    clip[0, :, placement] = 100 + torch.arange(63.0)
    clip[0, 1, shape] += 0.5  # the hand's shape changes between the two frames
    features, _ = encoder.inputs(clip)
    joints = encoder.joints(features)
    assert joints.shape == (1, 2, 21, 12)
    # Landmark 8 in the second frame: shape xyz, placement xyz, then the motion of each.
    assert joints[0, 1, 8].tolist() == [24.5, 25.5, 26.5, 124, 125, 126, 0.5, 0.5, 0.5, 0, 0, 0]


def test_graph_conv_is_optional_and_only_on_the_hands():
    torch.manual_seed(6)
    plain, graph = SignNet(classes=4), SignNet(classes=4, graph_conv=True)
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    # Per hand: 12·32 + 32, 32·32 + 32, then 21·32·120 + 120.
    assert count(graph) - count(plain) == 2 * 82_232
    assert not any(".graph." in name for name in plain.state_dict())
    assert {name.split(".graph.")[0] for name in graph.state_dict() if ".graph." in name} == {"regions.0", "regions.1"}
    with pytest.raises(ValueError, match="hand regions"):
        RegionEncoder((4,), width=40, graph=True)

    clip = torch.randn(2, CLIP_FRAMES, NORMALISED_SIZE)
    clip[..., PRESENCE] = 1.0
    clip[0, :, COORDINATE_SIZE:COORDINATE_SIZE + 4] = 0.0  # no hands in the first clip
    graph.eval()
    logits = graph(clip)
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
    left = graph.regions[0](clip)
    assert torch.equal(left[0], graph.regions[0].missing.expand(CLIP_FRAMES, -1))
    assert not torch.equal(left[1], plain.regions[0](clip)[1])


def test_a_model_with_graph_conv_exports_and_says_so(tmp_path):
    write_clips(tmp_path / "clips", per_sign=4)
    out = tmp_path / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=2, seed=0, test_signer="u2", runs=tmp_path / "runs",
                   graph_conv=True)
    assert report["graph_conv"] is True
    recogniser = load_recogniser(str(out))
    assert recogniser.info["graph_conv"] is True
    plain = train(tmp_path / "clips", tmp_path / "plain.onnx", epochs=1, seed=0, test_signer="u2",
                  runs=tmp_path / "runs")
    assert plain["graph_conv"] is False and "graph_conv" not in plain["info"]


def test_graph_conv_export_matches_pytorch_on_different_clips(tmp_path):
    torch.manual_seed(7)
    model = SignNet(classes=4, graph_conv=True).eval()
    rng = np.random.default_rng(7)
    probes = [to_model_input(fake_clip(rng, index)) for index in range(2)]
    out = tmp_path / "graph.onnx"
    export(model, out, ["A", "B", "C", IDLE_LABEL], None, probes[0])
    recogniser = load_recogniser(str(out))
    for probe in probes:
        with torch.no_grad():
            expected = torch.softmax(model(torch.from_numpy(probe)[None]), dim=1)[0].numpy()
        assert recogniser.predict(probe) == pytest.approx(expected, abs=1e-4)


def test_a_graph_model_starts_from_a_checkpoint_made_without_the_block(tmp_path):
    from checkpoint import load_into, save_checkpoint

    torch.manual_seed(8)
    labels = ["A", "B", IDLE_LABEL]
    source = SignNet(3)
    save_checkpoint(source, labels, "somewhere", tmp_path / "plain.pt")
    target = SignNet(3, graph_conv=True)
    fresh = target.regions[0].graph.out.weight.clone()
    assert load_into(target, labels, tmp_path / "plain.pt") == "somewhere"
    assert torch.equal(target.regions[0].encoder[0].weight, source.regions[0].encoder[0].weight)
    assert torch.equal(target.head[-1].weight, source.head[-1].weight)
    assert torch.equal(target.regions[0].graph.out.weight, fresh)


def test_ablation_can_measure_the_graph_conv(tmp_path):
    from ablate import ablate

    write_clips(tmp_path / "clips", per_sign=4)
    result = ablate(tmp_path / "clips", seeds=[0, 1], epochs=2, test_signer="u2", runs=tmp_path / "runs",
                    block="graph-conv")
    assert result["block"] == "graph-conv"
    assert set(result["variants"]) == {"with graph conv", "without graph conv"}
    paired = [a - b for a, b in zip(result["variants"]["with graph conv"]["top1"]["runs"],
                                    result["variants"]["without graph conv"]["top1"]["runs"])]
    assert result["difference"]["top1"]["runs"] == pytest.approx(paired)
    assert not list(tmp_path.rglob("*.onnx"))
