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

export function detect(trackers: Trackers, video: HTMLVideoElement, timeMs: number): Detection {
  const h = trackers.hands.detectForVideo(video, timeMs);
  const p = trackers.pose.detectForVideo(video, timeMs);
  const f = trackers.face.detectForVideo(video, timeMs);

  const out: Detection = { hands: h.landmarks };
  h.landmarks.forEach((points, i) => {
    // MediaPipe labels handedness as seen in the image it is given.
    // TODO: confirm against a live camera that "Left" is the signer's left hand.
    const label = h.handedness[i]?.[0]?.categoryName;
    if (label === "Left") out.leftHand = points;
    else if (label === "Right") out.rightHand = points;
  });
  out.pose = p.landmarks[0];
  out.blendshapes = f.faceBlendshapes?.[0]?.categories.map((c) => c.score);
  out.faceMatrix = f.facialTransformationMatrixes?.[0]?.data;
  return out;
}

export function closeTrackers(trackers: Trackers) {
  trackers.hands.close();
  trackers.pose.close();
  trackers.face.close();
}
