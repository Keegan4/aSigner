"""The sign model interface that scoring is written against.

The trained model comes from ml/ (handoff task T2). Until an exported model is
wired in here, `get_recogniser` returns None and practice reports that scoring
is unavailable. Tests replace it through FastAPI's dependency overrides.
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
    return None
