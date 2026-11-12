import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

import type { FrameParts } from "./features";

const MODELS = "https://storage.googleapis.com/mediapipe-models";

/** The three MediaPipe models; a tracker may load only some of them. */
export type Part = "hands" | "pose" | "face";
export const ALL_PARTS: readonly Part[] = ["hands", "pose", "face"];

export type Trackers = {
  hands?: HandLandmarker;
  pose?: PoseLandmarker;
  face?: FaceLandmarker;
  /** Whether the models run on the GPU, or fell back to the CPU. */
  delegate: "GPU" | "CPU";
  /** Run pose and face only on every nth call of `detect`. */
  every: { pose: number; face: number };
};

/** Milliseconds each model took on this call; a model that did not run is absent. */
export type Timings = Partial<Record<Part, number>>;

export type Detection = FrameParts & {
  pose?: NormalizedLandmark[];
  /** Every detected hand, for drawing. */
  hands: NormalizedLandmark[][];
  timings: Timings;
};

// Hands carry the sign and change fastest, so they are tracked on every frame.
// The body and face move slowly by comparison; tracking them less often roughly
// doubles the frame rate on machines where MediaPipe falls back to the CPU.
// The face model is the slowest of the three, and only its blendshapes and head
// angles are used, which change slowly, so it runs least often.
export const MAIN_THREAD_EVERY = { pose: 2, face: 4 };

// Once hands are found, MediaPipe follows them from frame to frame and skips the
// slow whole-image palm search while both hands keep passing these scores. A
// little below MediaPipe's 0.5 default, a hand that blurs mid-sign keeps being
// followed instead of being searched for again on the next frame.
export const HAND_TRACKING = { minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 };

// How much of the newest body position to take; the rest is the previous one.
// Steadies the arm, which the lightweight pose model places with visible jitter.
const POSE_SMOOTHING = 0.6;

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

const WASM = "/mediapipe/wasm";

/** Not exported by @mediapipe/tasks-vision, but accepted wherever it takes one. */
type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/**
 * The MediaPipe runtime files. In a module Web Worker, MediaPipe's own loader
 * fails: it tries importScripts, which module workers reject with an error it
 * does not expect. Importing the ES module build here sets the global
 * ModuleFactory it is looking for, and an empty loader path tells MediaPipe the
 * loader has already run.
 */
async function wasmFiles(useModule: boolean): Promise<WasmFileset> {
  if (!useModule) return FilesetResolver.forVisionTasks(WASM);
  const loader = new URL(`${WASM}/vision_wasm_module_internal.js`, self.location.origin).href;
  await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ loader);
  return { wasmLoaderPath: "", wasmBinaryPath: `${WASM}/vision_wasm_module_internal.wasm` };
}

async function create(
  parts: readonly Part[],
  delegate: "GPU" | "CPU",
  useModule: boolean,
): Promise<Omit<Trackers, "every">> {
  const fileset = await wasmFiles(useModule);
  const base = (path: string) => ({
    baseOptions: { modelAssetPath: `${MODELS}/${path}`, delegate },
    runningMode: "VIDEO" as const,
  });
  const [hands, pose, face] = await Promise.all([
    parts.includes("hands")
      ? HandLandmarker.createFromOptions(fileset, {
          ...base("hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
          numHands: 2,
          ...HAND_TRACKING,
        })
      : undefined,
    parts.includes("pose")
      ? PoseLandmarker.createFromOptions(fileset, {
          ...base("pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
          numPoses: 1,
        })
      : undefined,
    parts.includes("face")
      ? FaceLandmarker.createFromOptions(fileset, {
          ...base("face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        })
      : undefined,
  ]);
  return { hands, pose, face, delegate };
}

/**
 * Loads the given models, on the GPU when the browser allows it and on the CPU
 * otherwise. `useModule` loads MediaPipe's ES module build, which a module Web
 * Worker needs.
 */
export async function loadTrackers({
  parts = ALL_PARTS,
  every = MAIN_THREAD_EVERY,
  useModule = false,
}: { parts?: readonly Part[]; every?: Trackers["every"]; useModule?: boolean } = {}): Promise<Trackers> {
  quietenMediaPipeInfo();
  // Files that cannot be loaded at all would fail on the CPU too.
  await wasmFiles(useModule);
  try {
    return { ...(await create(parts, "GPU", useModule)), every };
  } catch (err) {
    console.warn("MediaPipe could not use the GPU here; tracking on the CPU instead.", err);
    return { ...(await create(parts, "CPU", useModule)), every };
  }
}

type Cache = {
  calls: number;
  pose?: NormalizedLandmark[];
  blendshapes?: number[];
  faceMatrix?: number[];
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

export function detect(
  trackers: Trackers,
  image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap,
  timeMs: number,
): Detection {
  let cache = caches.get(trackers);
  if (!cache) caches.set(trackers, (cache = { calls: 0 }));
  const call = cache.calls++;
  const timings: Timings = {};
  const timed = <T>(part: Part, run: () => T): T => {
    const start = performance.now();
    const result = run();
    timings[part] = performance.now() - start;
    return result;
  };

  const out: Detection = { hands: [], timings };
  if (trackers.hands) {
    const h = timed("hands", () => trackers.hands!.detectForVideo(image, timeMs));
    out.hands = h.landmarks;
    h.landmarks.forEach((points, i) => {
      // MediaPipe labels handedness as seen in the image it is given.
      // TODO: confirm against a live camera that "Left" is the signer's left hand.
      const label = h.handedness[i]?.[0]?.categoryName;
      if (label === "Left") out.leftHand = points;
      else if (label === "Right") out.rightHand = points;
    });
  }
  if (trackers.pose && call % trackers.every.pose === 0) {
    const p = timed("pose", () => trackers.pose!.detectForVideo(image, timeMs));
    cache.pose = smooth(cache.pose, p.landmarks[0]);
  }
  if (trackers.face && call % trackers.every.face === 0) {
    const f = timed("face", () => trackers.face!.detectForVideo(image, timeMs));
    cache.blendshapes = f.faceBlendshapes?.[0]?.categories.map((c) => c.score);
    const matrix = f.facialTransformationMatrixes?.[0]?.data;
    // A plain array, so the result can be posted out of a worker.
    cache.faceMatrix = matrix ? Array.from(matrix) : undefined;
  }
  out.pose = cache.pose;
  out.blendshapes = cache.blendshapes;
  out.faceMatrix = cache.faceMatrix;
  return out;
}

export function closeTrackers(trackers: Trackers) {
  trackers.hands?.close();
  trackers.pose?.close();
  trackers.face?.close();
}
