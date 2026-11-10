"""Loads the sign model exported by ml/train.py and adapts it to `Recogniser`.

The model takes a clip resampled to CLIP_FRAMES normalised frames, shape
(1, CLIP_FRAMES, NORMALISED_SIZE), and returns one logit per label. The labels
file and the weights sit beside the model: `signs.onnx`, `signs.onnx.data`
and `signs.labels.json`. Keep the three together.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import CLIP_FRAMES, NORMALISED_SIZE, resample


class OnnxRecogniser:
    def __init__(self, model_path: Path):
        labels_path = model_path.with_suffix(".labels.json")
        self.labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))
        # InferenceSession.run may be called from several threads at once.
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
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

    Reloads when the file changes, so a retrained model is picked up without a restart.
    """
    path = Path(model_path)
    if not path.is_file() or not path.with_suffix(".labels.json").is_file():
        return None
    return _load(str(path), path.stat().st_mtime)
