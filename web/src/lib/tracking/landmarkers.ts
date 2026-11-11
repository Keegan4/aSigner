import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

import type { FrameParts } from "./features";

const MODELS = "https://storage.googleapis.com/mediapipe-models";

export type Trackers = {
  hands: HandLandmarker;
  pose: PoseLandmarker;
  face: FaceLandmarker;
};

export type Detection = FrameParts & {
  pose?: NormalizedLandmark[];
  /** Every detected hand, for drawing. */
  hands: NormalizedLandmark[][];
};

let quietened = false;

/**
 * MediaPipe prints progress notes such as "INFO: Created TensorFlow Lite
 * XNNPACK delegate for CPU" through console.error, which the Next.js dev
 * overlay reports as an application error. Those lines are passed to
 * console.info instead; every other error is left alone.
 */
function quietenMediaPipeInfo() {
  if (quietened) return;
  quietened = true;
  const original = console.error;
  console.error = (...args: unknown[]) => {
    if (typeof args[0] === "string" && /^(INFO|W\d{4}|I\d{4})[: ]/.test(args[0])) {
      console.info(...args);
      return;
    }
    original(...args);
  };
}

export async function loadTrackers(): Promise<Trackers> {
  quietenMediaPipeInfo();
  const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  const base = (path: string) => ({
    baseOptions: { modelAssetPath: `${MODELS}/${path}`, delegate: "GPU" as const },
    runningMode: "VIDEO" as const,
  });
  const [hands, pose, face] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      ...base("hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
      numHands: 2,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      ...base("pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
      numPoses: 1,
    }),
    FaceLandmarker.createFromOptions(fileset, {
      ...base("face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    }),
  ]);
  return { hands, pose, face };
}

// Hands carry the sign and change fastest, so they are tracked on every frame.
// The body and face move slowly by comparison; tracking them less often roughly
// doubles the frame rate on machines where MediaPipe falls back to the CPU.
const POSE_EVERY = 2;
const FACE_EVERY = 3;
// How much of the newest body position to take; the rest is the previous one.
// Steadies the arm, which the lightweight pose model places with visible jitter.
const POSE_SMOOTHING = 0.6;

type Cache = {
  calls: number;
  pose?: NormalizedLandmark[];
  blendshapes?: number[];
  faceMatrix?: ArrayLike<number>;
};
const caches = new WeakMap<Trackers, Cache>();

function smooth(previous: NormalizedLandmark[] | undefined, next: NormalizedLandmark[] | undefined) {
  if (!previous || !next || previous.length !== next.length) return next;
  const keep = 1 - POSE_SMOOTHING;
  return next.map((point, i) => ({
    ...point,
    x: point.x * POSE_SMOOTHING + previous[i].x * keep,
    y: point.y * POSE_SMOOTHING + previous[i].y * keep,
    z: point.z * POSE_SMOOTHING + previous[i].z * keep,
  }));
}

export function detect(trackers: Trackers, video: HTMLVideoElement, timeMs: number): Detection {
  let cache = caches.get(trackers);
  if (!cache) caches.set(trackers, (cache = { calls: 0 }));
  const call = cache.calls++;

  const h = trackers.hands.detectForVideo(video, timeMs);
  if (call % POSE_EVERY === 0) {
    const p = trackers.pose.detectForVideo(video, timeMs);
    cache.pose = smooth(cache.pose, p.landmarks[0]);
  }
  if (call % FACE_EVERY === 0) {
    const f = trackers.face.detectForVideo(video, timeMs);
    cache.blendshapes = f.faceBlendshapes?.[0]?.categories.map((c) => c.score);
    cache.faceMatrix = f.facialTransformationMatrixes?.[0]?.data;
  }

  const out: Detection = { hands: h.landmarks };
  h.landmarks.forEach((points, i) => {
    // MediaPipe labels handedness as seen in the image it is given.
    // TODO: confirm against a live camera that "Left" is the signer's left hand.
    const label = h.handedness[i]?.[0]?.categoryName;
    if (label === "Left") out.leftHand = points;
    else if (label === "Right") out.rightHand = points;
  });
  out.pose = cache.pose;
  out.blendshapes = cache.blendshapes;
  out.faceMatrix = cache.faceMatrix;
  return out;
}

export function closeTrackers(trackers: Trackers) {
  trackers.hands.close();
  trackers.pose.close();
  trackers.face.close();
}
