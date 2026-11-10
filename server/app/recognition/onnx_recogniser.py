"""Loads the sign model exported by ml/train.py and adapts it to `Recogniser`.

The model takes a clip resampled to CLIP_FRAMES normalised frames, shape
(1, CLIP_FRAMES, NORMALISED_SIZE), and returns one logit per label. It is a
single file: the labels and the pass score travel inside it as ONNX metadata,
so a retrained model replaces the old one in one atomic rename.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.recognition.features import CLIP_FRAMES, NORMALISED_SIZE, resample

LABELS_KEY = "asigner.labels"  # JSON list of glosses, in output order
PASS_SCORE_KEY = "asigner.pass_score"  # optional float chosen at training time
INFO_KEY = "asigner.info"  # optional JSON object: how much data the model was trained on


class OnnxRecogniser:
    def __init__(self, model_path: Path):
        # InferenceSession.run may be called from several threads at once.
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        metadata = self._session.get_modelmeta().custom_metadata_map
        if LABELS_KEY not in metadata:
            raise ValueError(f"{model_path} carries no labels; export it with ml/train.py")
        self.labels: list[str] = json.loads(metadata[LABELS_KEY])
        self.pass_score: float | None = (
            float(metadata[PASS_SCORE_KEY]) if PASS_SCORE_KEY in metadata else None
        )
        # clips, signers, top1, test_clips, untested_signs, trained_at. Shown to
        # learners so a model trained on very little is visibly weak.
        self.info: dict | None = json.loads(metadata[INFO_KEY]) if INFO_KEY in metadata else None
        model_input = self._session.get_inputs()[0]
        if len(model_input.shape) != 3 or model_input.shape[-1] != NORMALISED_SIZE:
            raise ValueError(f"model must accept {NORMALISED_SIZE} features including presence flags; "
                             "retrain and re-export it with ml/train.py")
        self._input = model_input.name
        outputs = self._session.get_outputs()[0].shape[-1]
        if isinstance(outputs, int) and outputs != len(self.labels):
            raise ValueError(f"model has {outputs} outputs but {len(self.labels)} labels")

    def predict(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != NORMALISED_SIZE or len(frames) == 0:
            raise ValueError(f"expected (time, {NORMALISED_SIZE}) frames, got {frames.shape}")
        clip = resample(frames, CLIP_FRAMES)[None]
        logits = self._session.run(None, {self._input: clip})[0][0].astype(np.float64)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()


@lru_cache(maxsize=2)
def _load(path: str, modified: float) -> OnnxRecogniser:
    return OnnxRecogniser(Path(path))


def load_recogniser(model_path: str) -> OnnxRecogniser | None:
    """The model at `model_path`, or None if no model has been exported yet.

    Reloads when the file changes, so a retrained model is picked up by the
    next connection without a restart.
    """
    path = Path(model_path)
    if not path.is_file():
        return None
    return _load(str(path), path.stat().st_mtime)
