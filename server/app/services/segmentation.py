"""Turns a live stream of landmark frames into a run of sign glosses.

The model is trained on two-second clips of one sign, so it is shown the last
WINDOW_SECONDS of frames every STRIDE_SECONDS. A sign is taken as made when
the model names it with enough confidence several times running. A stretch of
"no sign" (the idle class, low confidence, or no hands in view) separates one
sign from the next, and a longer one ends the sentence.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.recognition.features import IDLE_LABEL, POSE, normalise
from app.recognition.recogniser import Recogniser
from app.services.practice import pass_score

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.2
MIN_WINDOW_SECONDS = 1.0  # no guesses until this much has been seen
MIN_WINDOW_FRAMES = 5
STABLE_PREDICTIONS = 3  # the same sign this many times running counts as signed
SENTENCE_GAP_SECONDS = 2.5  # this long without a sign ends the sentence
MAX_GLOSSES = 50
MAX_BUFFERED_FRAMES = 300


@dataclass
class Update:
    gloss: str | None = None
    """A sign that has just been recognised."""
    sentence: list[str] | None = None
    """The glosses of a sentence that has just ended."""


@dataclass
class Segmenter:
    recogniser: Recogniser
    glosses: list[str] = field(default_factory=list)
    _frames: deque = field(default_factory=deque)  # (time, normalised frame, hand seen)
    _last_prediction: float = float("-inf")
    _candidate: str | None = None
    _run: int = 0
    _emitted: str | None = None  # the sign currently held up, already counted
    _last_sign: float | None = None

    def push(self, raw: np.ndarray, at: float) -> Update:
        """Take one raw frame that arrived at time `at` (seconds, monotonic)."""
        self._frames.append((at, normalise(raw), bool(raw[:POSE].any())))
        while at - self._frames[0][0] > WINDOW_SECONDS or len(self._frames) > MAX_BUFFERED_FRAMES:
            self._frames.popleft()
        if at - self._last_prediction < STRIDE_SECONDS:
            return Update()
        self._last_prediction = at
        return self._step(self._label(at), at)

    def _label(self, at: float) -> str | None:
        """What the window shows: a gloss, or None for no sign."""
        if (
            len(self._frames) < MIN_WINDOW_FRAMES
            or at - self._frames[0][0] < MIN_WINDOW_SECONDS
            or not any(hand for _, _, hand in list(self._frames)[-MIN_WINDOW_FRAMES:])
        ):
            return None
        probabilities = np.asarray(
            self.recogniser.predict(np.stack([frame for _, frame, _ in self._frames]))
        )
        best = int(probabilities.argmax())
        label = self.recogniser.labels[best]
        if label == IDLE_LABEL or probabilities[best] < pass_score(self.recogniser):
            return None
        return label

    def _step(self, label: str | None, at: float) -> Update:
        if label != self._candidate:
            self._candidate, self._run = label, 0
        self._run += 1
        if self._run < STABLE_PREDICTIONS:
            return Update()

        if label is None:
            # A settled gap: the next sign counts even if it repeats the last one.
            self._emitted = None
            ended = (
                self.glosses
                and self._last_sign is not None
                and at - self._last_sign >= SENTENCE_GAP_SECONDS
            )
            return Update(sentence=self.flush()) if ended else Update()

        self._last_sign = at
        if label == self._emitted:
            return Update()
        self._emitted = label
        self.glosses.append(label)
        if len(self.glosses) >= MAX_GLOSSES:
            return Update(gloss=label, sentence=self.flush())
        return Update(gloss=label)

    def flush(self) -> list[str]:
        """End the sentence now and return its glosses (possibly none)."""
        glosses, self.glosses = self.glosses, []
        return glosses
