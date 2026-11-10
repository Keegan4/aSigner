"""Checks the pipeline end to end on made-up clips. Says nothing about real accuracy."""
import numpy as np

from data import augment, mirror, to_model_input
from train import train

from app.features import CLIP_FRAMES, FRAME_SIZE, LEFT_HAND, NORMALISED_SIZE, POSE, RIGHT_HAND
from app.onnx_recogniser import load_recogniser


def fake_clip(rng, label_index: int, frames: int = 50) -> np.ndarray:
    """A right hand whose shape depends on the label, with a body, jittered per clip."""
    shape = np.random.default_rng(100 + label_index).uniform(0.4, 0.6, 63)
    clip = np.zeros((frames, FRAME_SIZE), dtype=np.float32)
    clip[:, RIGHT_HAND:POSE] = shape + rng.normal(0, 0.004, (frames, 63))
    clip[:, POSE:POSE + 75] = np.random.default_rng(7).uniform(0.3, 0.7, 75)
    return clip


def write_clips(folder, signers=("u1", "u2"), per_sign=12):
    rng = np.random.default_rng(0)
    for index, label in enumerate("ABC"):
        (folder / label).mkdir(parents=True)
        for signer in signers:
            for n in range(per_sign):
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

    assert report["signs"] == ["A", "B", "C"]
    assert "u2 held out" in report["split"]
    assert report["top1"] > 0.9  # the made-up signs are easy to tell apart

    recogniser = load_recogniser(str(out))
    assert recogniser.labels == ["A", "B", "C"]
    probabilities = recogniser.predict(to_model_input(fake_clip(np.random.default_rng(3), 2)))
    assert recogniser.labels[int(probabilities.argmax())] == "C"
    assert list((tmp_path / "runs").glob("*/confusion.csv"))
