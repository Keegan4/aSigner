"""The sign model interface that scoring is written against.

The trained model comes from ml/ (handoff task T2). `get_recogniser` returns
None until a model has been exported to MODEL_PATH, and practice then reports
that scoring is unavailable. Tests replace it through FastAPI's dependency
overrides.
"""
import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Recogniser(Protocol):
    labels: list[str]
    """Sign glosses, in the order of the model's outputs."""

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Return one probability per label for a clip.

        `frames` has shape (time, NORMALISED_SIZE): the output of
        `features.normalise` for each frame, oldest first.
        """
        ...


class UnusableModel:
    """Stands in for a model file that is there but cannot be loaded.

    Most often an export from before the feature layout changed. Callers tell
    the learner scoring is unavailable instead of failing the connection.
    """

    labels: list[str] = []

    def __init__(self, reason: str):
        self.reason = reason


def get_recogniser() -> "Recogniser | UnusableModel | None":
    """The installed model, None when there is none, or UnusableModel when the file will not load."""
    from app.core.config import get_settings
    from app.recognition.onnx_recogniser import load_recogniser

    path = get_settings().model_path
    try:
        return load_recogniser(path)
    except Exception as exc:  # onnxruntime raises its own types for a bad file
        logger.warning("Sign model at %s cannot be used: %s", path, exc)
        return UnusableModel(str(exc))
