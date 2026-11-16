/**
 * A recorded take of one sign, as written by ml/export_demos.py into
 * public/demos. Coordinates are MediaPipe's image-normalised ones with x
 * stretched so both axes share a unit; y grows downwards and may run past 1
 * for parts below the camera's view. Points are flattened as x, y, z.
 */
export type DemoClip = {
  label: string;
  /** Frames a second the take was tracked at. */
  fps: number;
  /** Pose landmarks 0-24 for every frame. */
  pose: number[][];
  /** The signer's own left and right hands: 21 points, or null where untracked. */
  left: (number[] | null)[];
  right: (number[] | null)[];
};

export type Point = { x: number; y: number; z: number };
export type Figure = { pose: Point[]; left: Point[] | null; right: Point[] | null };
/** The part of the clip's space to show: its centre and size. */
export type Box = { cx: number; cy: number; width: number; height: number };

const L_SHOULDER = 11;
const R_SHOULDER = 12;

const cache = new Map<string, Promise<DemoClip>>();

/** Fetch a demo clip once per page, however many figures show it. */
export function loadDemo(path: string): Promise<DemoClip> {
  let clip = cache.get(path);
  if (!clip) {
    clip = fetch(path).then((res) => {
      if (!res.ok) throw new Error(`${path} returned ${res.status}`);
      return res.json() as Promise<DemoClip>;
    });
    clip.catch(() => cache.delete(path));
    cache.set(path, clip);
  }
  return clip;
}

/** Seconds from the first frame to the last. */
export function duration(clip: DemoClip): number {
  return Math.max(0, clip.pose.length - 1) / clip.fps;
}

function points(a: number[], b: number[], share: number): Point[] {
  const out: Point[] = [];
  for (let i = 0; i < a.length; i += 3) {
    out.push({
      x: a[i] + (b[i] - a[i]) * share,
      y: a[i + 1] + (b[i + 1] - a[i + 1]) * share,
      z: a[i + 2] + (b[i + 2] - a[i + 2]) * share,
    });
  }
  return out;
}

/**
 * The figure `seconds` into the clip, blended between the two recorded frames
 * around that moment so a take tracked at 10 frames a second still moves
 * smoothly. A hand missing from one of the two frames is taken from the nearer.
 */
export function figureAt(clip: DemoClip, seconds: number): Figure {
  const last = clip.pose.length - 1;
  const at = Math.min(Math.max(seconds * clip.fps, 0), last);
  const a = Math.floor(at);
  const b = Math.min(a + 1, last);
  const share = at - a;
  const hand = (rows: (number[] | null)[]) => {
    const from = rows[a];
    const to = rows[b];
    if (from && to) return points(from, to, share);
    const nearer = share < 0.5 ? from : to;
    return nearer ? points(nearer, nearer, 0) : null;
  };
  return { pose: points(clip.pose[a], clip.pose[b], share), left: hand(clip.left), right: hand(clip.right) };
}

/**
 * Where to point the view for a whole clip: head and chest, widened to keep
 * every tracked hand in shot. Fixed per clip so the figure does not drift.
 */
export function frameBox(clip: DemoClip): Box {
  let mx = 0;
  let my = 0;
  let shoulders = 0;
  for (const pose of clip.pose) {
    const lx = pose[L_SHOULDER * 3];
    const ly = pose[L_SHOULDER * 3 + 1];
    const rx = pose[R_SHOULDER * 3];
    const ry = pose[R_SHOULDER * 3 + 1];
    mx += (lx + rx) / 2;
    my += (ly + ry) / 2;
    shoulders += Math.hypot(lx - rx, ly - ry);
  }
  const frames = Math.max(1, clip.pose.length);
  mx /= frames;
  my /= frames;
  const unit = shoulders / frames || 0.5;

  let minX = mx - 1.2 * unit;
  let maxX = mx + 1.2 * unit;
  let minY = my - 1.25 * unit;
  let maxY = my + 1.15 * unit;
  for (const rows of [clip.left, clip.right]) {
    for (const hand of rows) {
      if (!hand) continue;
      for (let i = 0; i < hand.length; i += 3) {
        minX = Math.min(minX, hand[i] - 0.15 * unit);
        maxX = Math.max(maxX, hand[i] + 0.15 * unit);
        minY = Math.min(minY, hand[i + 1] - 0.15 * unit);
        maxY = Math.max(maxY, hand[i + 1] + 0.15 * unit);
      }
    }
  }
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, width: maxX - minX, height: maxY - minY };
}
