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

## Review and implementation notes

Implemented as planned, with these additions found in review:

- `ml/test_train.py` also hard-coded the hand slices (`:63`, `63:126`, `126:252`) in the
  `RegionEncoder` grouping test; it now uses `PART_WIDTHS`.
- README's parameter count changed: `2,004,000 + 161 × C` (was `1,980,000 + 161 × C`).
- `THUMB_TIP`, `FINGER_JOINTS` and `HAND_DETAIL` are public in `features.py`, so the
  separability script and tests name the same points the normaliser uses.
- **Aspect ratio:** x and y are image-normalised (x by width, y by height) and z is roughly in
  x's units, so on a 640×480 camera the palm frame is built in slightly stretched space. Rotation
  invariance is therefore approximate on real data (exact in the tests, which rotate the raw
  values). Fixing it needs the frame's aspect ratio, which the 256-float frame does not carry;
  left as is.
- Synthetic check (`ml/test_separability.py`): fists that differ only in thumb depth, turned at
  random in 3D, with noise. Leave-one-out nearest-centroid accuracy was about 40–60% on the
  original 63 values and 83–95% on the 75 palm-frame values; the top feature was always
  `point 4 palm-n` (effect size about 8–9). Nearest-centroid weights all features equally, so
  its accuracy is a floor; the effect-size list is the sharper signal.
- `test_practice.py` and `test_translate.py` had fake recognisers reading the right hand at a
  hard-coded offset (63); they now use `NORMALISED_PARTS[1].start`.

### Measured on the owner's clips (2026-11-12)

15 clips each of S and M, all from one signer (`u1`):

| Feature set | Leave-one-out accuracy |
|---|---|
| Original hand shape (63) | 96.7% |
| Palm frame and thumb distances (75) | 96.7% |
| Both (138) | 96.7% |

S and M already separate on the original features, and the latest training run
(`runs/20261111-005308`) got its 3 held-out S and 3 held-out M clips right. What separates them
is mostly the pinky and the knuckles' height (`point 20 palm-n`, `point 18 y`, `point 9 y`),
i.e. how the hand was held, not the thumb: `point 4 palm-n` ranks 68th of 138 (effect 1.1,
against about 8 in the synthetic test). So on these recordings MediaPipe does not place the
hidden thumb clearly differently for the two letters, which is the limit this plan warned about.
Other pairs: palm-frame values helped M/N (86.7% → 96.7%) and A/S (92.9% → 100%), not T/N
(100% → 91.4%). On the public ASLNow samples (83 S, 88 M) they did not help (82.5% → 77.2%).

With one signer and six held-out clips this cannot show whether the change helps a new signer or
a new hand angle. A second signer's S and M clips would.

### Model comparison on the owner's clips (2026-11-12)

Two models per seed trained on the same 331 clips (no pretraining, 80 epochs): one with the old
features (`main`), one with this branch. Each was scored on the same 79 held-out clips, as recorded and with
each hand turned in 3D about its wrist by 20° and 40° (4 random axes per clip, identical for
both models). The turn is applied to the raw image coordinates, so it is a stand-in for real
hand angles: it does not model occlusion or how MediaPipe tracks a turned hand, and it favours
features built to ignore rotation.

| Hand turned | Seed | Old: all letters | New: all letters | Old: S M N A E T | New: S M N A E T |
|---|---|---|---|---|---|
| 0° (as recorded) | 0 | 91.1% | 89.9% | 94.4% | 94.4% |
| 0° (as recorded) | 1 | 89.9% | 94.9% | 72.2% | 94.4% |
| 20° | 0 | 75.0% | 85.4% | 65.3% | 88.9% |
| 20° | 1 | 73.1% | 85.1% | 51.4% | 76.4% |
| 40° | 0 | 41.1% | 69.0% | 37.5% | 70.8% |
| 40° | 1 | 49.1% | 69.0% | 45.8% | 59.7% |

As recorded, the two are level within noise (3 held-out clips per letter). With the hand turned,
the new features hold up clearly better in both seeds. S and M themselves: as recorded both
models got 3/3 of each; turned 20°, S went from 8/12 and 6/12 (old) to 11/12 and 9/12 (new), while M
went from 8/12 and 5/12 to 11/12 and 5/12. The remaining fist mistakes are mostly among S, M and N.
