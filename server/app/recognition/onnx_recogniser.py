"""Loads the sign model exported by ml/train.py and adapts it to `Recogniser`.

The model takes a clip resampled to CLIP_FRAMES normalised frames, shape
(1, CLIP_FRAMES, NORMALISED_SIZE), and returns one logit per label. It is a
single file: the labels and the pass score travel inside it as ONNX metadata,
so a retrained model replaces the old one in one atomic rename.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.recognition.features import CLIP_FRAMES, NORMALISED_SIZE, resample

logger = logging.getLogger(__name__)

LABELS_KEY = "asigner.labels"  # JSON list of glosses, in output order
PASS_SCORE_KEY = "asigner.pass_score"  # optional float chosen at training time
INFO_KEY = "asigner.info"  # optional JSON object: how much data the model was trained on

CPU = "CPUExecutionProvider"
# GPU execution providers in order of preference. Each exists only in the
# matching onnxruntime package: onnxruntime-gpu (CUDA), the ROCm/MIGraphX
# builds (AMD), onnxruntime-directml (any DirectX 12 GPU on Windows),
# onnxruntime-openvino (Intel) and the standard macOS package (CoreML).
GPU_PROVIDERS = (
    ("CUDAExecutionProvider", {}),
    ("ROCMExecutionProvider", {}),
    ("MIGraphXExecutionProvider", {}),
    ("DmlExecutionProvider", {}),
    ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
    ("CoreMLExecutionProvider", {}),
)


def pick_providers() -> list:
    """The first GPU provider this onnxruntime offers, then the CPU for anything it cannot run."""
    available = set(ort.get_available_providers())
    gpu = [(name, options) for name, options in GPU_PROVIDERS if name in available][:1]
    return [*gpu, CPU]


def _session(model_path: Path, providers: list | None) -> ort.InferenceSession:
    """A session on the chosen providers, or on the CPU if a GPU provider will not start."""
    chosen = pick_providers() if providers is None else providers
    try:
        return ort.InferenceSession(str(model_path), providers=chosen)
    except Exception as exc:  # a missing driver or library surfaces here, in onnxruntime's own types
        if chosen == [CPU]:
            raise
        name = chosen[0][0] if isinstance(chosen[0], tuple) else chosen[0]
        logger.warning("Running the sign model on the CPU: %s would not start (%s)", name, exc)
        return ort.InferenceSession(str(model_path), providers=[CPU])


class OnnxRecogniser:
    def __init__(self, model_path: Path, providers: list | None = None):
        """`providers` defaults to a GPU when onnxruntime offers one (see pick_providers)."""
        # InferenceSession.run may be called from several threads at once.
        self._session = _session(model_path, providers)
        # The provider that actually took the model, e.g. "CUDAExecutionProvider".
        self.provider: str = self._session.get_providers()[0]
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
    recogniser = OnnxRecogniser(Path(path))
    logger.info("Loaded the sign model at %s on %s", path, recogniser.provider)
    return recogniser


def load_recogniser(model_path: str) -> OnnxRecogniser | None:
    """The model at `model_path`, or None if there is no usable model.

    Reloads when the file changes, so a retrained model is picked up by the
    next connection without a restart. A file that cannot be used (for example
    one trained before the feature layout changed) is reported in the log and
    treated as no model, so practice says scoring is unavailable instead of
    the connection failing.
    """
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        return _load(str(path), path.stat().st_mtime)
    except Exception as exc:  # onnxruntime raises its own types for corrupt files
        logger.warning("Ignoring the sign model at %s: %s", path, exc)
        return None
