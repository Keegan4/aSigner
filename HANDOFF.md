# Handoff

For whoever picks this up next, human or AI. Read this and [README.md](README.md) first. Questions and task claims go on the noticeboard PR (see [NOTICEBOARD.md](NOTICEBOARD.md)).

## What aSigner is

A portfolio project (not commercial) that teaches Singapore Sign Language (SgSL). The learner performs a sign on camera and a recognition model gives feedback. A sign-to-English translator is the secondary mode. Scope is A–Z fingerspelling plus about 50 signs.

## Decisions already made (do not reopen without the owner)

- **Landmarks, not video.** The browser runs MediaPipe (hands, pose, face) and sends one 256-float frame per video frame over a WebSocket. Video never leaves the browser and is never stored.
- **Inference on the server.** FastAPI serves an ONNX model on CPU.
- **One normaliser.** `server/app/features.py` is imported by both serving and training. Do not write a second one.
- **Local LLM only.** The translator's sentence layer uses Ollama (`SENTENCE_MODEL`), with no external LLM API. It must fall back to plain glosses when Ollama is down.
- **Google sign-in, Postgres.** Only text is stored: progress, attempts, past translations.
- **Sign Bank media is not in the repo.** It is copyright SADeaf, all rights reserved. Link to it; do not copy from it unless written permission arrives.
- **No unreviewed teaching content.** Sign descriptions stay empty until an SgSL signer has checked them.

## Current state

Working and tested (29 server tests on SQLite and on Postgres 16; web type-check, lint and build clean):

- Auth, session cookie, lessons API, translation history API, delete-my-data, authenticated `/ws/echo`.
- Schema and one Alembic migration (applied to Postgres 16; `alembic check` reports no drift); seed with the Alphabet lesson.
- Home page and `/check` camera page.
- Frame layout (`web/src/lib/features.ts` and `server/app/features.py`, which must match) and normaliser.
- Practice scoring (T3): `/ws/practice`, spaced repetition, `GET /api/progress`. Scoring runs against the `Recogniser` interface in `server/app/recogniser.py`; no model is installed, so the live socket answers `no_model` until T2 delivers one.
- Lesson, practice, quiz and progress screens (T4). Driven in headless Chromium with a fake camera and a fake model: an attempt was scored, stored and shown on the progress page.
- Translator groundwork (T5): `server/app/sentence.py` asks Ollama for a sentence and falls back to the glosses; `POST /api/translations` stores one; `/translations` lists them with delete and browser speech. Tested against a fake Ollama only.
- Docker (T6): both images build and `postgres`, `server` and `web` start and serve. The `ollama` service was not started.

Never verified:

- Live camera tracking with a person in view, and frame rate on real hardware.
- Whether MediaPipe's "Left"/"Right" labels match the signer's hands (`web/src/lib/landmarkers.ts` has a TODO).
- A real Google sign-in round trip.
- The sentence layer against a real Ollama model, and how good its sentences are.
- The GitHub Actions workflow: it runs on pull requests and pushes to `main`, and has not run yet.
- The "Name the sign" quiz with real demo media (no sign has a `demo_path`; it is disabled until one does).

## Where things are

| Path | Contents |
|---|---|
| `server/app/main.py` | App wiring and session middleware |
| `server/app/auth.py` | Google OAuth |
| `server/app/ws.py` | WebSocket auth (`authorise`), echo and `/ws/practice` (message format in its docstring) |
| `server/app/practice.py` | Scoring thresholds and review schedule |
| `server/app/recogniser.py` | The model interface T2 must implement |
| `server/app/features.py` | Frame decode and normalise |
| `server/app/models.py` | All tables |
| `server/seed/lessons.yaml` | Lesson content |
| `server/tests/` | pytest; `conftest.py` has the signed-in client fixture |
| `web/src/components/CameraTracker.tsx` | Camera, tracking loop, overlay |
| `web/src/components/PracticeSession.tsx` | Camera plus one scored attempt; used by practice and quiz |
| `web/src/lib/` | Frame packing, MediaPipe setup, API helpers, practice socket (and its mock) |

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

### T1. Sample recorder
A signed-in page where the user picks a sign, records a few seconds, and the landmark clip (frames only) is saved with its label. Decide with the owner whether clips go to the server or download as files. Output format should be a `.npy`/`.npz` array of raw 256-float frames plus label and signer id.
Done when: a recorded clip can be loaded in Python and passed through `normalise`.

### T2. Training pipeline (`ml/`)
Dataset loader, augmentation (mirror, scale, rotation, time-warp, jitter, frame drop, face-feature dropout), a small Transformer or GRU over normalised frames, signer-held-out evaluation, ONNX export. Start with A–Z. Public ASL fingerspelling data can be used for pretraining because SgSL uses the same one-handed alphabet.
Depends on: T1 for SgSL data. Done when: an exported `.onnx` plus a report with top-1/top-5 accuracy and a confusion matrix.

### T3. Practice scoring and progress (server): done
A `/ws/practice` endpoint: client names the target sign, server buffers frames, runs the model, returns score, pass/fail and the closest confusion, then writes `attempts` and updates `progress` (spaced repetition: review date moves out on success, resets on failure).
Can start now against a stub model interface. Done when: tests replay recorded frames and assert the stored attempt and progress.

### T4. Lesson and practice UI (web): done, apart from what needs demo media
Lesson page (sign list, demo slot, description, Sign Bank link), practice screen using `CameraTracker`, quiz in both directions, progress dashboard.
Can start now against the existing lessons API and a mocked practice socket.

### T5. Translator mode
Sign segmentation (idle and transition detection, confidence threshold), gloss buffer, Ollama client with fallback, saving to `translations`, a history page with delete, browser speech output.
Done: Ollama client with fallback, storage, history page with delete, speech output. Left: segmentation, the gloss buffer and the live translator screen, which depend on T2 for a real model. The live screen should build the gloss list on the server from the model's output and reuse `make_sentence`.

### T6. Infrastructure: done, apart from a first CI run and the `ollama` service
Run `docker compose up --build` and fix what breaks, run the migration on Postgres, add GitHub Actions for pytest, type-check and lint.
Can start now.

### T7. Human-only tasks (owner)
- Create the Google OAuth client and `.env`.
- Open `/check` in a normal browser and confirm tracking, frame rate and hand labelling.
- Email `sgsl_linguistics@sadeaf.org.sg` to ask permission to use Sign Bank material for a portfolio project.
- Choose the ~50-sign vocabulary and get descriptions reviewed by an SgSL signer.
- Record reference and training clips.

## Running and testing

See [README.md](README.md). Before opening a PR: `pytest` in `server/`, and `npx next typegen`, `npx tsc --noEmit` and `npm run lint` in `web/`.
