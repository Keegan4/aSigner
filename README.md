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
- Video translator (`/translate/video`): translate a recorded video instead of the live camera, with every model window's top guesses shown and downloadable as JSON. A debugging aid for the model. The video is tracked in the browser; only landmarks and their video times are sent, and sentences are only saved when asked.

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

Training (`train.py` and `pretrain.py`) uses a GPU whenever PyTorch can see one, and the CPU otherwise; it prints which at the start. In order: CUDA (NVIDIA, or AMD on a [ROCm build](https://pytorch.org/get-started/locally/), which PyTorch also reports as `cuda`), XPU (Intel Arc and Core Ultra graphics, on an [XPU build](https://pytorch.org/docs/stable/notes/get_start_xpu.html)), then MPS (Apple Silicon). Each needs the matching PyTorch build; the default one sees only the CPU (and MPS on a Mac). Integrated GPUs other than Intel Core Ultra's are mostly not supported by these builds. For an NVIDIA GPU, install a CUDA build of PyTorch instead (leave out `--extra-index-url ...cpu`, or follow [pytorch.org](https://pytorch.org/get-started/locally/)); `python -c "import torch; print(torch.cuda.is_available())"` should print `True`. On an Apple Silicon Mac the standard macOS build already includes MPS (`torch.backends.mps.is_available()` should print `True`); if an operation is missing there, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run it on the CPU. For a model this small a GPU may not be much faster, since augmentation runs on the CPU either way. The exported model always runs on the CPU, as the server does: GPU inference was tried and was slower for this model.

This writes `server/models/signs.onnx` (one file, with the labels and pass score inside it), which the server loads without a restart, plus accuracy and a confusion matrix under `ml/runs/`. With clips from one person only, the reported accuracy is optimistic; record a second person to get an honest number.

The model takes 32 frames of 538 features: 532 normalised values plus six validity flags for left/right hand shape, left/right hand placement, pose and face/head. Each hand shape is 138 values: the 21 points relative to the wrist in units of palm length (63), the same points in the palm's own frame, so they do not change when the hand turns (63), and the distances from the thumb tip to the middle joint, end joint and tip of each finger (12). In the palm frame one axis always points out of the palm, which is what separates closed fists such as S (thumb in front of the fingers) from M (thumb behind them). The browser sends x (and MediaPipe's z, which shares x's scale) in units of image height, so x and y measure the same distance on a 640x480 or 16:9 camera and a hand turned in the picture keeps the same palm-frame values. Clips saved before this have no `x_units` field and are converted on load assuming the recorder's 640x480 image; `ml/video_to_clip.py` uses each video's real size. These flags are inferred from existing recordings; the browser's 256-value frame format is unchanged. Face presence is inferred from nonzero face/head values, so a completely neutral face cannot be distinguished from missing face tracking.

Separate left-hand, right-hand, pose and face/head encoders each produce 80 dimensions from positions, frame differences and validity flags. Differences are zero for the first frame and whenever either adjacent observation is invalid, preventing tracking loss/recovery from appearing as motion. They describe motion over resampled clip time, not physical velocity. Entirely missing regions use learned embeddings. The four outputs are concatenated into 320 dimensions and processed by a residual temporal convolution (depthwise kernel 3, then a pointwise projection).

The model then prepends a learned classification token and adds learned positional embeddings. Two Transformer encoder layers, each with 20 attention heads (16 dimensions per head), let every token attend to the entire clip. The classification token's output passes through layer normalisation and a three-layer classifier (`320 → 320 → 160 → labels`, including the no-sign class). Each hidden classifier layer uses batch normalisation, GELU and dropout. All dropout probabilities are 0.1. Training requires at least two clips after the signer split; inference accepts a single clip using the saved batch-normalisation statistics.

There are `2,004,000 + 161 × C` trainable parameters, where `C` includes the no-sign class (2,020,261 for 100 signs plus no-sign). Existing recordings remain usable, but models exported for an earlier input size (382 or 388 features) must be retrained and re-exported; the server rejects them at load time with a "needs retraining" message. A pretraining checkpoint from before the change is refused too: run `pretrain.py` again.

### Letters that look alike

Several letters are closed fists that differ only in where the thumb sits (S/M, M/N, A/S, E/S, T/N). To check whether your recordings hold the difference at all, before training:

```bash
cd ml
.venv/Scripts/python separability.py S M
```

It prints how well the two signs separate on the original hand shape and on the palm-frame values, and which values differ most. If the palm-frame line stays near 50%, MediaPipe is not placing the hidden thumb differently for the two letters, and no retraining will fix it. When recording these letters, vary how the hand is turned so the model learns the shape rather than one angle. Then retrain (`pretrain.py` first if you use it, then `train.py`) and check those letters' rows in `runs/<time>/confusion.csv`.

### Pretraining on public data (optional)

With few clips of your own, the model can start from hand shapes learned on a public fingerspelling dataset ([ASLNow!](https://huggingface.co/datasets/sid220/asl-now-fingerspelling), MIT licence: 2,122 single-frame hand landmark samples, A to Z):

```bash
cd ml
git clone https://huggingface.co/datasets/sid220/asl-now-fingerspelling data/public/asl-now-fingerspelling
python pretrain.py
python train.py --init runs/pretrained.pt
```

The public samples are still hands with no body or movement, so pretraining teaches hand shapes only, and the checkpoint is never served; `train.py` still trains on your clips afterwards. A model started this way records it, in its training details.

## Tests

```bash
cd server
.venv/Scripts/python -m pytest
```

To check the ML architecture and ONNX export without running training or optimizer steps, run from the repository root with the ML dependencies installed:

```bash
python -m pytest -q ml/test_train.py -k 'not train_exports and not training_ and not report_names'
```

## Data and credit

Sign references follow the [SgSL Sign Bank](https://blogs.ntu.edu.sg/sgslsignbank/) (copyright The Singapore Association for the Deaf). No Sign Bank media is included in this repository. Lesson descriptions are intentionally empty until they have been reviewed by an SgSL signer.
