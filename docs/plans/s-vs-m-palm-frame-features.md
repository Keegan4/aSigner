# Telling S from M: palm-frame hand features

## Context

S and M are both closed fists. Seen as 2D landmarks they nearly coincide; what separates them is
whether the thumb lies in front of the three fingers (S) or behind them (M), plus how tightly
those fingers curl.

That information is already in the data: every frame carries MediaPipe's `z` for all 21 hand
points, and `normalise()` keeps it. The problem is how it reaches the model:

- `hand_shape()` in `server/app/recognition/features.py:84` is wrist-relative and palm-scaled but
  **not rotation-normalised**. "Thumb in front of the fingers" is therefore a different
  combination of x, y and z for every hand orientation, and the model has to learn it from about
  25 clips per sign.
- There are **no explicit geometric features** (no distances, no depth ordering). The first layer
  of each `RegionEncoder` is linear, so it cannot form a distance on its own.

So the workaround is to hand the model the answer directly: express the hand in the palm's own
coordinate frame, where "in front of / behind the fingers" is always the same axis, and add
thumb-to-finger distances. The browser's 256-float frame and the stored clips stay as they are;
existing recordings are re-used, only retraining is needed.

**Limit to be honest about:** this works only if MediaPipe places the hidden thumb differently
for the two letters. If it reports the same landmarks for both, no feature can separate them.
Step 1 measures this on the real clips before and after, so we know rather than hope.

Not recommended now (larger changes, kept as fallbacks): switching to MediaPipe `worldLandmarks`
(changes the frame format, invalidates recorded clips and the public pretraining data), or a
second-stage classifier on the hand image (video never leaves the browser by design).

## Changes

### 1. Separability check — new `ml/separability.py`

A small script, since no clips are in the repo and the owner has to run it locally:

    python separability.py --clips ../server/data/clips S M

- Loads clips with `load_clips` (`ml/data.py:40`), normalises each frame with `normalise`, takes
  the mean hand-shape part of the signing hand per clip.
- Reports leave-one-out nearest-centroid accuracy for the pair (numpy only, no new dependency),
  once on the old 63 hand-shape values and once on the full new hand part, plus the five most
  separating features by standardised mean difference.
- Works for any pair (M/N, A/S, E/S, T/N have the same problem).

### 2. Palm-frame features — `server/app/recognition/features.py`

Add `hand_detail(points, left)` next to `hand_shape`, returning 75 values per hand, and append
it to the hand-shape part:

- Start from the wrist-relative, palm-scaled points. For the left hand flip x first, so a
  mirrored hand gives exactly the same values as the original.
- Palm frame: `u` = wrist → middle knuckle (9); `v` = pinky knuckle (17) → index knuckle (5),
  made orthogonal to `u`; `n = u × v` (palm normal). If `v` is degenerate, return zeros.
- **63 values:** all 21 points in `(u, v, n)` coordinates — the hand shape independent of how the
  hand is turned. The `n` coordinate of the thumb tip against the finger joints is the
  S-versus-M signal.
- **12 values:** distance from the thumb tip (4) to the middle joint, end joint and tip of each
  of the four fingers (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20).
- `PART_WIDTHS` becomes `(138, 138, 63, 63, 75, 55)`. Everything else (`NORMALISED_PARTS`,
  `COORDINATE_SIZE`, `PRESENCE`, `NORMALISED_SIZE` = 538, `RegionEncoder` input widths in
  `ml/train.py:62`) is derived from it. The six flags are unchanged: detail shares the
  hand-shape flag. The original 63 orientation-dependent values stay, because orientation
  itself distinguishes other signs.

Update the module docstring and the size comments.

### 3. Tests

- `server/tests/test_features.py`: replace the hard-coded `388` and the `63:126` / `189:252`
  slices with `NORMALISED_SIZE` / `NORMALISED_PARTS`; add tests that
  - detail is unchanged when the hand is rotated in 3D (the original shape values do change),
  - a thumb tip moved across the fingers along the palm normal changes the sign of its `n`
    coordinate relative to the finger joints,
  - a missing or collapsed hand gives zeros.
- `ml/test_train.py`: add a test that `mirror()` of a right-hand clip produces the same detail
  values in the left-hand slot.
- `server/tests/test_onnx_recogniser.py:66`: the error-message match uses `NORMALISED_SIZE`
  instead of the literal `388`.

### 4. Model file and docs

- The committed `server/models/signs.onnx` takes 388 features and will be reported as "needs
  retraining" by the existing check (`onnx_recogniser.py:40`); the README already says it is not
  a usable model. Pretrained checkpoints are refused by `load_into` with its existing
  "run pretrain.py again" message. No code change needed for either.
- `README.md:80`: update the feature count and describe the palm-frame values. Add a short
  "confusable letters" note with the `separability.py` command and the retrain steps
  (`pretrain.py`, then `train.py`), and a recording tip: record S and M at several hand turns.

## Verification

- `cd server && pytest` and `python -m pytest -q ml/test_train.py ml/test_pretrain.py`.
- Run `ml/separability.py` on synthetic S-like and M-like hands (same 2D layout, thumb depth
  flipped, random 3D rotation) and confirm the new features separate them where the old do not.
- Train on a synthetic set end to end (`ml/test_train.py` already does a small run) and export,
  confirming the server loads a 538-feature model.
- Cannot be done here (no clips in the repo): the owner runs `separability.py S M` on their
  clips, retrains, and checks the S/M cells of `runs/<time>/confusion.csv`. If separability stays
  near 50% after the change, MediaPipe is not resolving the thumb and the fallback is
  `worldLandmarks` or more varied recordings.
