# aSigner

Learn Singapore Sign Language (SgSL) by signing to your camera and getting live feedback. A student portfolio project, not a substitute for learning from Deaf instructors.

## Status

Built so far:

- FastAPI server: Google sign-in, session cookie, lessons API, translation history API, delete-my-data, authenticated WebSocket.
- Database schema and migrations (users, signs, lessons, attempts, progress, translations).
- Next.js app: home page with lessons, camera check page with hand, body and face tracking.
- Landmark frame format (256 floats per frame) and the shared normaliser.

Not built yet: the sign recognition model, practice scoring, quiz, spaced repetition, and the translator.

## How it works

```
Browser: camera -> MediaPipe hands + pose + face -> 1 KB landmark frame
   | WebSocket                      | REST via Next.js proxy (/api)
FastAPI: normalise -> sign model -> scoring / captions -> Postgres
```

Video never leaves the browser. Only landmark coordinates are sent, and only text (progress and past translations) is stored.

## Run locally

Server (uses a local SQLite file unless `DATABASE_URL` is set):

```bash
cd server
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m app.seed
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Web:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000.

### Google sign-in

1. In Google Cloud Console, create an OAuth client of type "Web application".
2. Add the redirect URI `http://localhost:3000/api/auth/callback`.
3. Copy `.env.example` to `.env` and fill in `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and a long random `SESSION_SECRET`.

### Docker

`docker compose up --build` starts Postgres, Ollama, the server and the web app. This path has not been run yet.

## Tests

```bash
cd server
.venv/Scripts/python -m pytest
```

## Data and credit

Sign references follow the [SgSL Sign Bank](https://blogs.ntu.edu.sg/sgslsignbank/) (copyright The Singapore Association for the Deaf). No Sign Bank media is included in this repository. Lesson descriptions are intentionally empty until they have been reviewed by an SgSL signer.
