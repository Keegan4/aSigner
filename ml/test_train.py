"""Checks the pipeline end to end on made-up clips. Says nothing about real accuracy."""
import numpy as np

from data import augment, mirror, to_model_input, transition
from train import threshold_curve, train

from app.features import CLIP_FRAMES, FRAME_SIZE, LEFT_HAND, NORMALISED_SIZE, POSE, RIGHT_HAND
from app.onnx_recogniser import load_recogniser


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


def test_train_exports_a_model_the_server_can_load(tmp_path):
    write_clips(tmp_path / "clips")
    out = tmp_path / "models" / "signs.onnx"
    report = train(tmp_path / "clips", out, epochs=30, seed=0, test_signer="u2", runs=tmp_path / "runs")

    assert report["labels"] == ["A", "B", "C", "_idle"]  # no-sign is always a class
    assert "u2 held out" in report["split"]
    assert report["signs_without_test_clips"] == []
    assert report["top1"] > 0.9  # the made-up signs are easy to tell apart
    assert len(report["threshold_curve"]) == 14
    assert 0.3 <= report["pass_score"] <= 0.95  # 36 held-out clips is enough to choose one

    # One file, nothing left half-written beside it.
    assert [p.name for p in out.parent.iterdir()] == ["signs.onnx"]
    recogniser = load_recogniser(str(out))
    assert recogniser.labels == ["A", "B", "C", "_idle"]
    assert recogniser.pass_score == report["pass_score"]
    probabilities = recogniser.predict(to_model_input(fake_clip(np.random.default_rng(3), 2)))
    assert recogniser.labels[int(probabilities.argmax())] == "C"
    assert list((tmp_path / "runs").glob("*/confusion.csv"))


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
