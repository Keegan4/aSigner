import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from app.recognition.features import (
    CLIP_FRAMES,
    FRAME_SIZE,
    NORMALISED_SIZE,
    normalise,
    resample,
)
from app.recognition.onnx_recogniser import INFO_KEY, LABELS_KEY, PASS_SCORE_KEY, OnnxRecogniser, load_recogniser
from app.services.practice import score_clip

LABELS = ["A", "B", "C"]


def write_model(folder, labels=LABELS, outputs=None, pass_score=None, input_size=NORMALISED_SIZE,
                info=None):
    """A tiny stand-in for the trained model: mean over time, then a fixed linear layer.

    Class k responds to feature k, so a clip whose feature 1 is large scores as "B".
    """
    outputs = outputs or len(labels)
    weights = np.zeros((input_size, outputs), dtype=np.float32)
    for k in range(outputs):
        weights[k, k] = 10.0
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["clip"], ["mean"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["mean", "weights"], ["logits"]),
        ],
        "stand_in",
        [helper.make_tensor_value_info("clip", TensorProto.FLOAT, [1, CLIP_FRAMES, input_size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, outputs])],
        [helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.ravel())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if labels is not None:
        model.metadata_props.add(key=LABELS_KEY, value=json.dumps(labels))
    if pass_score is not None:
        model.metadata_props.add(key=PASS_SCORE_KEY, value=str(pass_score))
    if info is not None:
        model.metadata_props.add(key=INFO_KEY, value=json.dumps(info))
    path = folder / "signs.onnx"
    onnx.save(model, path)
    return path


def clip(feature: int, time: int) -> np.ndarray:
    frames = np.zeros((time, NORMALISED_SIZE), dtype=np.float32)
    frames[:, feature] = 1.0
    return frames


def test_missing_model_means_no_recogniser(tmp_path):
    assert load_recogniser(str(tmp_path / "signs.onnx")) is None


def test_legacy_model_is_rejected_at_load_with_reexport_guidance(tmp_path):
    path = write_model(tmp_path, input_size=382)
    with pytest.raises(ValueError, match=f"{NORMALISED_SIZE} features.*retrain and re-export"):
        OnnxRecogniser(path)


def test_unusable_model_counts_as_no_model_and_is_logged(tmp_path, caplog):
    """A stale or corrupt file must not take the practice socket down with it."""
    stale = write_model(tmp_path, input_size=382)
    with caplog.at_level("WARNING"):
        assert load_recogniser(str(stale)) is None
    assert "retrain and re-export" in caplog.text

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "signs.onnx").write_bytes(b"not a model")
    assert load_recogniser(str(corrupt / "signs.onnx")) is None


def test_predicts_probabilities_for_any_clip_length(tmp_path):
    recogniser = load_recogniser(str(write_model(tmp_path)))
    assert recogniser.labels == LABELS
    for time in (1, 7, CLIP_FRAMES, 300):
        probabilities = recogniser.predict(clip(feature=1, time=time))
        assert probabilities.shape == (3,)
        assert probabilities.sum() == pytest.approx(1.0)
        assert LABELS[int(probabilities.argmax())] == "B"


def test_rejects_wrong_shapes_and_label_mismatch(tmp_path):
    recogniser = load_recogniser(str(write_model(tmp_path)))
    with pytest.raises(ValueError):
        recogniser.predict(np.zeros((4, FRAME_SIZE), dtype=np.float32))  # raw, not normalised
    with pytest.raises(ValueError):
        recogniser.predict(np.zeros((0, NORMALISED_SIZE), dtype=np.float32))

    for name, labels in (("mismatch", ["A", "B"]), ("unlabelled", None)):
        folder = tmp_path / name
        folder.mkdir()
        with pytest.raises(ValueError):
            OnnxRecogniser(write_model(folder, labels=labels, outputs=3))


def test_pass_score_travels_with_the_model(tmp_path):
    """Scoring reads `recogniser.pass_score` and falls back to its default when it is None."""
    plain = tmp_path / "plain"
    tuned = tmp_path / "tuned"
    plain.mkdir()
    tuned.mkdir()
    assert load_recogniser(str(write_model(plain))).pass_score is None
    assert load_recogniser(str(write_model(tuned, pass_score=0.75))).pass_score == 0.75


def test_works_with_practice_scoring(tmp_path):
    """The contract task T3 scores against: raw frames in, a Score out."""
    recogniser = load_recogniser(str(write_model(tmp_path)))
    rng = np.random.default_rng(0)
    raw = [rng.uniform(0.2, 0.8, FRAME_SIZE).astype(np.float32) for _ in range(20)]
    result = score_clip(recogniser, raw, "A")
    assert 0.0 <= result.score <= 1.0
    assert result.passed or result.closest in LABELS


def test_resample_keeps_order_and_endpoints():
    frames = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, NORMALISED_SIZE), np.float32)
    out = resample(frames, 4)
    assert out.shape == (4, NORMALISED_SIZE)
    assert out[0, 0] == 0 and out[-1, 0] == 9
    assert (np.diff(out[:, 0]) >= 0).all()
    assert resample(frames[:1], 4).shape == (4, NORMALISED_SIZE)
    assert np.isfinite(normalise(np.zeros(FRAME_SIZE, np.float32))).all()


def test_training_info_travels_with_the_model(tmp_path):
    plain = tmp_path / "plain"
    described = tmp_path / "described"
    plain.mkdir()
    described.mkdir()
    info = {"clips": 240, "signers": 2, "top1": 0.91, "test_clips": 60, "trained_at": "2026-11-10T08:00:00+00:00"}
    assert load_recogniser(str(write_model(plain))).info is None
    assert load_recogniser(str(write_model(described, info=info))).info == info
