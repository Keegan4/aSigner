import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from app.features import CLIP_FRAMES, FRAME_SIZE, NORMALISED_SIZE, normalise, resample
from app.onnx_recogniser import load_recogniser
from app.practice import score_clip

LABELS = ["A", "B", "C"]


def write_model(folder, labels=LABELS, outputs=None):
    """A tiny stand-in for the trained model: mean over time, then a fixed linear layer.

    Class k responds to feature k, so a clip whose feature 1 is large scores as "B".
    """
    outputs = outputs or len(labels)
    weights = np.zeros((NORMALISED_SIZE, outputs), dtype=np.float32)
    for k in range(outputs):
        weights[k, k] = 10.0
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["clip"], ["mean"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["mean", "weights"], ["logits"]),
        ],
        "stand_in",
        [helper.make_tensor_value_info("clip", TensorProto.FLOAT, [1, CLIP_FRAMES, NORMALISED_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, outputs])],
        [helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.ravel())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = folder / "signs.onnx"
    onnx.save(model, path)
    path.with_suffix(".labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return path


def clip(feature: int, time: int) -> np.ndarray:
    frames = np.zeros((time, NORMALISED_SIZE), dtype=np.float32)
    frames[:, feature] = 1.0
    return frames


def test_missing_model_means_no_recogniser(tmp_path):
    assert load_recogniser(str(tmp_path / "signs.onnx")) is None


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

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError):
        load_recogniser(str(write_model(other, labels=["A", "B"], outputs=3)))


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
