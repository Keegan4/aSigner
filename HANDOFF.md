# Handoff

For whoever picks this up next, human or AI. Read this and [README.md](README.md) first. Questions and task claims go on the noticeboard PR (see [NOTICEBOARD.md](NOTICEBOARD.md)).

## What aSigner is

A portfolio project (not commercial) that teaches Singapore Sign Language (SgSL). The learner performs a sign on camera and a recognition model gives feedback. A sign-to-English translator is the secondary mode. Scope is A–Z fingerspelling plus about 50 signs.

## Decisions already made (do not reopen without the owner)

- **Landmarks, not video.** The browser runs MediaPipe (hands, pose, face) and sends one 256-float frame per video frame over a WebSocket. Video never leaves the browser and is never stored.
- **Inference on the server.** FastAPI serves an ONNX model on CPU. GPU execution providers were tried (PR #17, reverted): for this small model at one clip per prediction, the CPU was faster.
- **One normaliser.** `server/app/recognition/features.py` is imported by both serving and training. Do not write a second one.
- **Local LLM only.** The translator's sentence layer uses Ollama (`SENTENCE_MODEL`), with no external LLM API. It must fall back to plain glosses when Ollama is down.
- **Google sign-in, Postgres.** Only text is stored: progress, attempts, past translations.
- **Sign Bank media is not in the repo.** It is copyright SADeaf, all rights reserved. Link to it; do not copy from it unless written permission arrives.
- **No unreviewed teaching content.** Sign descriptions stay empty until an SgSL signer has checked them.

## Current state

Working and tested (server and training tests, web type-check, lint and build; CI runs all of it on every pull request, the server tests on SQLite and Postgres 16):

- Auth, session cookie, lessons API, translation history API, delete-my-data, authenticated `/ws/echo`.
- Schema and one Alembic migration (applied to Postgres 16; `alembic check` reports no drift); seed with the Alphabet lesson.
- Home page and `/check` camera page.
- Frame layout (`web/src/lib/tracking/features.ts` and `server/app/recognition/features.py`, which must match) and normaliser.
- Practice scoring (T3): `/ws/practice`, spaced repetition, `GET /api/progress`. Scoring runs against the `Recogniser` interface in `server/app/recognition/recogniser.py`; no model is installed, so the live socket answers `no_model` until T2 delivers one.
- Lesson, practice, quiz and progress screens (T4). Driven in headless Chromium with a fake camera and a fake model: an attempt was scored, stored and shown on the progress page.
- Translator groundwork (T5): `server/app/services/sentence.py` asks Ollama for a sentence and falls back to the glosses; `POST /api/translations` stores one; `/translations` lists them with delete and browser speech. Tested against a fake Ollama only.
- Live translator (T5): `server/app/services/segmentation.py` segments a frame stream into glosses, `/ws/translate` stores each finished sentence, `/translate` shows it. Tested with a fake model only; the page was loaded in headless Chromium but no sign was ever recognised in a browser.
- Model honesty: practice and translate show what the loaded model was trained on, with a caution when that is one signer, few held-out clips or untested signs (`server/app/services/model_info.py`). A model file that will not load (for example one exported before the feature layout changed) gives a "needs retraining" message instead of a dead socket.
- Docker (T6): both images build and `postgres`, `server` and `web` start and serve. The `ollama` service was not started.

Never verified:

- Live camera tracking with a person in view, and frame rate on real hardware.
- Whether MediaPipe's "Left"/"Right" labels match the signer's hands (`web/src/lib/tracking/landmarkers.ts` has a TODO).
- A real Google sign-in round trip.
- The sentence layer against a real Ollama model, and how good its sentences are.
- The "Name the sign" quiz with real demo media (no sign has a `demo_path`; it is disabled until one does).

## Where things are

| Path | Contents |
|---|---|
| `server/app/main.py` | App wiring and session middleware |
| `server/app/core/` | Settings (`config.py`) and database session (`db.py`) |
| `server/app/models.py` | All tables |
| `server/app/recognition/` | Frame decode and normalise (`features.py`), the model interface (`recogniser.py`), the ONNX loader (`onnx_recogniser.py`). Imports nothing but numpy and onnxruntime, because `ml/` imports it too |
| `server/app/services/` | Logic with no HTTP in it: scoring and review schedule (`practice.py`), segmentation for the translator (`segmentation.py`), the Ollama sentence layer (`sentence.py`), clip storage (`clips.py`), saving translations (`translations.py`) |
| `server/app/routes/` | REST endpoints, one module per resource, including Google OAuth (`auth.py`) and the signed-in user dependency (`deps.py`) |
| `server/app/ws/` | WebSocket endpoints, one module per socket (message formats in their docstrings); `session.py` has `authorise` |
| `server/seed/lessons.yaml` | Lesson content |
| `server/tests/` | pytest; `conftest.py` has the signed-in client fixture |
| `ml/` | Training: `data.py` (loading, augmentation), `train.py` (model, evaluation, export), `pull_samples.py` (takes from the training page's database into clip files) |
| `deploy/training-vm/` | Docker Compose and Caddy for running the training page on its own Linux VM |
| `web/src/app/` | Routes only; each page renders one component |
| `web/src/components/<feature>/` | `camera` (tracker, check page), `home`, `lessons` (lesson, quiz), `practice`, `progress`, `record`, `train` (training page), `translate` (live and history), `ui` (shared page frame and small parts) |
| `web/src/lib/` | `api.ts` (types and fetch helper), `practice.ts` (practice socket and its mock), `tracking/` (frame packing, MediaPipe setup) |
| `web/src/hooks/` | Shared data hooks |

### Where new code goes

- A new REST resource: a module in `server/app/routes/` with its own `router`, added to the list in `routes/__init__.py`. A new socket: the same in `server/app/ws/`.
- Routes and sockets stay thin. Anything worth testing without HTTP goes in `server/app/services/`. Services do not import from `routes/` or `ws/`.
- Anything the model or training needs goes in `server/app/recognition/` and must not import FastAPI, settings or the database at module level.
- A new screen: a page in `web/src/app/` and a component in its feature folder under `web/src/components/`. Server imports are absolute (`from app.services.practice import ...`); web imports use the `@/` alias.

## Gotchas

- **Next.js is 16.4.** It differs from older versions; read the docs in `web/node_modules/next/dist/docs/` before changing framework-level code.
- **WebSockets do not go through the Next proxy.** The browser connects straight to the server (`NEXT_PUBLIC_WS_URL`). The cookie still applies because cookies ignore ports. The server checks the `Origin` header against `FRONTEND_URL`.
- **Close codes need an accepted socket.** `authorise` accepts before closing with 4401 so the browser can see the code.
- **Python.** The server was developed on 3.14. PyTorch and MediaPipe's Python package may not support 3.14; check before starting `ml/` and use 3.12 or 3.13 there if needed.
- **Route types are generated.** `npx tsc --noEmit` fails on `LayoutProps` in a fresh checkout until `npx next typegen` (or a build) has run.
- **Dynamic routes need Suspense.** `cacheComponents` is on, so a page that reads `useParams`/`useSearchParams` must sit inside `<Suspense>` or the build fails.
- **Tests on Postgres.** Set `TEST_DATABASE_URL` to an empty database; every table in it is dropped for each test.
- **Trying the practice UI without a model.** Build the web app with `NEXT_PUBLIC_MOCK_PRACTICE=1`: results are made up and nothing is stored.

## Open tasks

Each task can be done independently unless it lists a dependency. Claim one on the noticeboard before starting so two people do not build the same thing.

### T1. Sample recorder: done
`/record` page and `/api/clips` (`server/app/routes/clips.py`, `server/app/services/clips.py`). One `.npz` per clip under `server/data/clips/<label>/`: raw frames `(time, 256)`, `label`, an opaque `signer` id, `fps`. A "No sign" class (`_idle`) is recorded the same way. Clips are deleted with the account. `DEV_LOGIN=true` gives a localhost-only sign-in so clips can be recorded without Google OAuth.
Verified with a live camera by the owner.

### T2. Training pipeline (`ml/`): done in code, no usable model yet
`ml/train.py` loads clips, augments them (mirror, roll, scale, jitter, time crop, frame drop, face dropout, windows that straddle two signs), trains, holds out a whole signer when there is more than one, and writes `report.json` (top-1, top-5, pass-score curve, signs with no held-out clip) and `confusion.csv` under `ml/runs/`. It exports one file, `server/models/signs.onnx`, with labels, pass score and training info in its metadata, moved into place atomically; the server picks it up on the next connection.
First real numbers (10 Nov 2026, 47 clips, one signer, about one training clip per letter): 40-45% top-1 on 20 held-out clips over three seeds. That is a pipeline check, not a usable model.
Still needed: about 25 clips per sign, "No sign" clips, and a second signer so accuracy is measured on someone the model has not seen.

### T3. Practice scoring and progress (server): done
A `/ws/practice` endpoint: client names the target sign, server buffers frames, runs the model, returns score, pass/fail and the closest confusion, then writes `attempts` and updates `progress` (spaced repetition: review date moves out on success, resets on failure).
Can start now against a stub model interface. Done when: tests replay recorded frames and assert the stored attempt and progress.

### T4. Lesson and practice UI (web): done, apart from what needs demo media
Lesson page (sign list, demo slot, description, Sign Bank link), practice screen using `CameraTracker`, quiz in both directions, progress dashboard.
Can start now against the existing lessons API and a mocked practice socket.

### T5. Translator mode: built, untested with a real model
Sign segmentation (idle and transition detection, confidence threshold), gloss buffer, Ollama client with fallback, saving to `translations`, a history page with delete, browser speech output.
Done in code: Ollama client with fallback, storage, history page with delete, speech output, segmentation (`server/app/services/segmentation.py`), `/ws/translate` and the `/translate` screen. Segmentation shows the model the last 2 s of frames every 0.2 s; its constants were chosen without a trained model and need tuning against real signing once one exists.

### T6. Infrastructure: done, apart from the `ollama` service, which has not been started
Run `docker compose up --build` and fix what breaks, run the migration on Postgres, add GitHub Actions for pytest, type-check and lint.
Can start now.

### T8. Training page on a VM: built, not yet deployed
`/train` (`web/src/components/train/Trainer.tsx`) and `/api/training` (`server/app/routes/training.py`, `server/app/services/training.py`). Takes go into Postgres (`training_samples`: raw frames as bytea, label, frame count, fps, anonymous contributor hash, time); signs added on the page go into `training_signs`. No Google sign-in: a shared `TRAINING_ACCESS_CODE`, required unless `FRONTEND_URL` is localhost. `ml/pull_samples.py` syncs the takes into clip files that `train.py` reads. `deploy/training-vm/` runs Postgres, server, web and Caddy (HTTPS, needed for the camera) on a Linux VM.
Tested: server endpoints (pytest, SQLite), the pull script against made-up rows, and the page's code, sign list, add and remove flows in a browser against a stubbed API. Not yet: a real take recorded through `/train`, the Docker stack on a VM, or a pull from a real Postgres.

### T7. Human-only tasks (owner)
- Create the Google OAuth client and `.env`.
- Open `/check` in a normal browser and confirm tracking, frame rate and hand labelling.
- Email `sgsl_linguistics@sadeaf.org.sg` to ask permission to use Sign Bank material for a portfolio project.
- Choose the ~50-sign vocabulary and get descriptions reviewed by an SgSL signer.
- Record training clips on `/record`: about 25 per sign plus "No sign" takes, and get a second person to record a set.
- Record reference (demo) clips for each sign.
- Say how doubled letters are fingerspelled in SgSL (the translator only repeats a letter after a pause).
- Decide whether a failed attempt resets a sign's mastery to 0 or steps it down one level.

## Running and testing

See [README.md](README.md). Before opening a PR: `pytest` in `server/`, and `npx next typegen`, `npx tsc --noEmit` and `npm run lint` in `web/`.
