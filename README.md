# aSigner

Learn Singapore Sign Language (SgSL) by signing to your camera and getting live feedback. A student portfolio project, not a substitute for learning from Deaf instructors.

## Status

Built so far:

- FastAPI server: Google sign-in, session cookie, lessons API, translation history API, delete-my-data, authenticated WebSocket.
- Database schema and migrations (users, signs, lessons, attempts, progress, translations).
- Next.js app: home page with lessons, camera check page with hand, body and face tracking.
- Landmark frame format (256 floats per frame) and the shared normaliser.
- Practice scoring over a WebSocket, spaced repetition and a progress API, waiting on the model.
- Lesson, practice, quiz and progress screens.
- Translator: live sign segmentation over a WebSocket, local Ollama sentence layer with a plain-gloss fallback, and a history page with delete and speech. Waiting on the model.

Not there yet: a trained sign recognition model. Until clips are recorded and a model is trained, practice cannot score real attempts and the translator cannot recognise signs.

## How it works

```
Browser: camera -> MediaPipe hands + pose + face -> 1 KB landmark frame
   | WebSocket                      | REST via Next.js proxy (/api)
FastAPI: normalise -> sign model -> scoring / captions -> Postgres
```

Video never leaves the browser. Only landmark coordinates are sent. Progress and past translations are stored as text. Training clips recorded on `/record` are stored as landmark coordinates (no video) until you delete them or your account.

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

`docker compose up --build` starts Postgres, Ollama, the server and the web app. The `ollama` service has not been run yet.

## Training the sign model

1. Add `DEV_LOGIN=true` to `.env` to sign in locally without Google (ignored unless `FRONTEND_URL` is localhost), or set up Google sign-in.
2. Open `/record` and record takes for each sign. Aim for 25 per sign, and vary distance, position and hand angle. Clips are saved under `server/data/clips/`.
3. Train and export:

```bash
cd ml
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python train.py
```

This writes `server/models/signs.onnx` (one file, with the labels and pass score inside it), which the server loads without a restart, plus accuracy and a confusion matrix under `ml/runs/`. With clips from one person only, the reported accuracy is optimistic; record a second person to get an honest number.

The model takes 32 frames of 382 normalised features. It projects each frame to 320 dimensions, prepends a learned classification token, and adds learned positional embeddings. Two Transformer encoder layers, each with 20 attention heads (16 dimensions per head), let every token attend to the entire clip. The classification token's output passes through layer normalisation and a three-layer classifier (`320 → 320 → 160 → labels`, including the no-sign class). Each hidden classifier layer uses batch normalisation, GELU and dropout. All dropout probabilities are 0.1. Training requires at least two clips after the signer split; inference accepts a single clip using the saved batch-normalisation statistics.

## Tests

```bash
cd server
.venv/Scripts/python -m pytest
```

## Data and credit

Sign references follow the [SgSL Sign Bank](https://blogs.ntu.edu.sg/sgslsignbank/) (copyright The Singapore Association for the Deaf). No Sign Bank media is included in this repository. Lesson descriptions are intentionally empty until they have been reviewed by an SgSL signer.
