"""The sign model interface that scoring is written against.

The trained model comes from ml/ (handoff task T2). `get_recogniser` returns
None until a model has been exported to MODEL_PATH, and practice then reports
that scoring is unavailable. Tests replace it through FastAPI's dependency
overrides.
"""
from typing import Protocol

import numpy as np


class Recogniser(Protocol):
    labels: list[str]
    """Sign glosses, in the order of the model's outputs."""

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Return one probability per label for a clip.

        `frames` has shape (time, NORMALISED_SIZE): the output of
        `features.normalise` for each frame, oldest first.
        """
        ...


def get_recogniser() -> Recogniser | None:
    from app.core.config import get_settings
    from app.recognition.onnx_recogniser import load_recogniser

    return load_recogniser(get_settings().model_path)
