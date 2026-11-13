"""Turn a video or animated GIF of one sign into a training clip.

    python video_to_clip.py hello.gif --label HELLO --signer ref1

Runs the same three MediaPipe models the browser runs (hands, pose, face) and
writes the same .npz the /record page produces, so the result can sit beside
recorded clips under server/data/clips/. Needs its own environment, because
MediaPipe's Python package lags behind new Python versions:

    py -3.13 -m venv .venv-mp
    .venv-mp/Scripts/python -m pip install mediapipe pillow

Only use footage you have the right to use. Clips made from someone else's
video are derived from it: keep them out of the repository.
"""
import argparse
import sys
import urllib.request
import uuid
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.recognition.features import (  # noqa: E402
    BLENDSHAPES, FACE, FRAME_SIZE, HAND_POINTS, HEAD, LEFT_HAND, POSE, POSE_POINTS, RIGHT_HAND, X_UNITS,
)
from data import to_height_units  # noqa: E402

# The same model files web/src/lib/tracking/landmarkers.ts loads.
MODELS = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "hand": "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face": "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
CACHE = Path(__file__).resolve().parent / ".cache"


def model(name: str) -> str:
    path = CACHE / Path(FILES[name]).name
    if not path.is_file():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{MODELS}/{FILES[name]}", path)
    return str(path)


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """RGB frames and frames per second."""
    if path.suffix.lower() == ".gif":
        from PIL import Image, ImageSequence

        image = Image.open(path)
        frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(image)]
        return frames, 1000.0 / (image.info.get("duration") or 100)
    import cv2

    video = cv2.VideoCapture(str(path))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames, fps


def points(landmarks, count: int) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks[:count]], dtype=np.float32).ravel()


def extract(frames: list[np.ndarray], fps: float) -> np.ndarray:
    """Raw landmark frames, laid out exactly as web/src/lib/tracking/features.ts packs them.

    Like the browser, x and z are in units of image height.
    """
    video = vision.RunningMode.VIDEO
    base = lambda name: tasks.BaseOptions(model_asset_path=model(name))  # noqa: E731
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base("hand"), running_mode=video, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=base("pose"), running_mode=video, num_poses=1))
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base("face"), running_mode=video, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    out = np.zeros((len(frames), FRAME_SIZE), dtype=np.float32)
    for index, rgb in enumerate(frames):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        at = int(index * 1000 / fps)
        h = hands.detect_for_video(image, at)
        for landmarks, handed in zip(h.hand_landmarks, h.handedness):
            start = {"Left": LEFT_HAND, "Right": RIGHT_HAND}.get(handed[0].category_name)
            if start is not None:
                out[index, start:start + HAND_POINTS * 3] = points(landmarks, HAND_POINTS)
        p = pose.detect_for_video(image, at)
        if p.pose_landmarks:
            out[index, POSE:FACE] = points(p.pose_landmarks[0], POSE_POINTS)
        f = face.detect_for_video(image, at)
        if f.face_blendshapes:
            scores = [c.score for c in f.face_blendshapes[0]][:BLENDSHAPES]
            out[index, FACE:FACE + len(scores)] = scores
        if f.facial_transformation_matrixes:
            m = np.asarray(f.facial_transformation_matrixes[0])
            out[index, HEAD:HEAD + 3] = (
                np.arcsin(np.clip(-m[2, 0], -1, 1)), np.arctan2(m[2, 1], m[2, 2]), np.arctan2(m[1, 0], m[0, 0]),
            )
    for tracker in (hands, pose, face):
        tracker.close()
    if frames:
        height, width = frames[0].shape[:2]
        out = to_height_units(out, width / height)
    return out


def convert(source: Path, label: str, signer: str, clips_dir: Path, trim: bool = True) -> dict:
    frames, fps = read_frames(source)
    raw = extract(frames, fps)
    total = len(raw)
    hand_seen = raw[:, LEFT_HAND:POSE].any(axis=1)
    if trim and hand_seen.any():
        # Reference videos start and end at rest; keep the signing, plus a short lead-in and tail.
        seen = np.flatnonzero(hand_seen)
        margin = max(1, int(round(fps * 0.2)))
        raw = raw[max(0, seen[0] - margin) : seen[-1] + margin + 1]
        hand_seen = raw[:, LEFT_HAND:POSE].any(axis=1)
    summary = {
        "source_frames": total, "frames": len(raw), "fps": round(fps, 1), "seconds": round(len(raw) / fps, 1),
        "hand_seen": float(hand_seen.mean()) if len(raw) else 0.0,
        "pose_seen": float(raw[:, POSE:FACE].any(axis=1).mean()) if len(raw) else 0.0,
        "face_seen": float(raw[:, FACE:HEAD].any(axis=1).mean()) if len(raw) else 0.0,
    }
    folder = clips_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{signer}_{uuid.uuid4().hex}.npz"
    np.savez_compressed(path, frames=raw, label=label, signer=signer, fps=np.float32(fps), x_units=X_UNITS)
    return summary | {"path": str(path)}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--label", required=True, help="the sign's gloss, as in the lessons")
    parser.add_argument("--signer", required=True, help="an opaque id for the person in the video")
    parser.add_argument("--clips", type=Path, default=here.parent / "server" / "data" / "clips")
    parser.add_argument("--no-trim", action="store_true", help="keep the frames before and after the hands appear")
    args = parser.parse_args()
    print(convert(args.source, args.label, args.signer, args.clips, trim=not args.no_trim))
