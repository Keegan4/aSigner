/**
 * Per-frame feature layout sent to the server. Must match ml/features.py.
 *
 * Values are raw MediaPipe outputs (image-normalised x, y and relative z).
 * Normalisation (centring, scaling) happens once, in Python, so training
 * and live inference cannot drift apart. Missing parts are all zeros.
 */
export const HAND_POINTS = 21;
export const POSE_POINTS = 25; // MediaPipe pose landmarks 0-24: head, arms, hips
export const BLENDSHAPES = 52;

export const LEFT_HAND = 0;
export const RIGHT_HAND = LEFT_HAND + HAND_POINTS * 3; // 63
export const POSE = RIGHT_HAND + HAND_POINTS * 3; // 126
export const FACE = POSE + POSE_POINTS * 3; // 201
export const HEAD = FACE + BLENDSHAPES; // 253: yaw, pitch, roll in radians
export const FRAME_SIZE = HEAD + 3; // 256 floats = 1024 bytes

export type Point = { x: number; y: number; z: number };

export type FrameParts = {
  leftHand?: Point[];
  rightHand?: Point[];
  pose?: Point[];
  blendshapes?: number[];
  /** Column-major 4x4 face transformation matrix from MediaPipe. */
  faceMatrix?: ArrayLike<number>;
};

function writePoints(out: Float32Array, offset: number, points: Point[] | undefined, count: number) {
  if (!points) return;
  for (let i = 0; i < Math.min(count, points.length); i++) {
    out[offset + i * 3] = points[i].x;
    out[offset + i * 3 + 1] = points[i].y;
    out[offset + i * 3 + 2] = points[i].z;
  }
}

/** Yaw, pitch, roll from the rotation part of a column-major 4x4 matrix. */
export function headAngles(m: ArrayLike<number>): [number, number, number] {
  const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
  const yaw = Math.asin(Math.max(-1, Math.min(1, -r20)));
  const pitch = Math.atan2(r21, r22);
  const roll = Math.atan2(r10, r00);
  return [yaw, pitch, roll];
}

export function packFrame(parts: FrameParts): Float32Array {
  const out = new Float32Array(FRAME_SIZE);
  writePoints(out, LEFT_HAND, parts.leftHand, HAND_POINTS);
  writePoints(out, RIGHT_HAND, parts.rightHand, HAND_POINTS);
  writePoints(out, POSE, parts.pose, POSE_POINTS);
  if (parts.blendshapes) out.set(parts.blendshapes.slice(0, BLENDSHAPES), FACE);
  if (parts.faceMatrix) out.set(headAngles(parts.faceMatrix), HEAD);
  return out;
}
