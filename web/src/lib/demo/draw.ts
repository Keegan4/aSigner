import type { Box, Figure, Point } from "./clip";

/** Fixed colours: the figure sits on the same dark panel as the camera view. */
const BODY = "#a3a3a3";
const EDGE = "#171717";
const HAND = "#f59e0b"; // the camera overlay's hand colour

type Px = { x: number; y: number };

const NOSE = 0;
const L_EYE = 2;
const R_EYE = 5;
const L_EAR = 7;
const R_EAR = 8;
const ARMS = [
  { shoulder: 11, elbow: 13, wrist: 15, hand: "left" },
  { shoulder: 12, elbow: 14, wrist: 16, hand: "right" },
] as const;

/** The palm outline, then each finger from its knuckle to its tip (MediaPipe hand indices). */
const PALM = [0, 1, 5, 9, 13, 17];
const FINGERS = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

const between = (a: Px, b: Px): Px => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const distance = (a: Px, b: Px) => Math.hypot(a.x - b.x, a.y - b.y);

function trace(ctx: CanvasRenderingContext2D, path: Px[], closed: boolean) {
  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  for (const p of path.slice(1)) ctx.lineTo(p.x, p.y);
  if (closed) ctx.closePath();
}

/**
 * A rounded limb or filled shape with a dark rim, so that an arm crossing the
 * chest or a finger crossing the palm stays readable.
 */
function shape(
  ctx: CanvasRenderingContext2D,
  path: Px[],
  width: number,
  colour: string,
  rim: number,
  closed = false,
) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  trace(ctx, path, closed);
  ctx.strokeStyle = EDGE;
  ctx.lineWidth = width + rim * 2;
  ctx.stroke();
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.stroke();
  if (closed) {
    ctx.fillStyle = colour;
    ctx.fill();
  }
}

function dot(ctx: CanvasRenderingContext2D, at: Px, radius: number, colour: string) {
  ctx.beginPath();
  ctx.arc(at.x, at.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
}

function drawHand(ctx: CanvasRenderingContext2D, hand: Point[], px: Px[], rim: number) {
  const width = Math.max(2, distance(px[0], px[9]) * 0.2);
  const depth = (indices: number[]) => indices.reduce((sum, i) => sum + hand[i].z, 0) / indices.length;
  // Farthest part first, so fingers curled in front of the palm cover it and
  // fingers behind it are covered. MediaPipe's z shrinks towards the camera.
  const parts = [
    { indices: PALM, closed: true },
    ...FINGERS.map((indices) => ({ indices, closed: false })),
  ].sort((a, b) => depth(b.indices) - depth(a.indices));
  for (const part of parts) {
    shape(ctx, part.indices.map((i) => px[i]), width, HAND, rim, part.closed);
  }
}

/** Draw one moment of a clip, fitted to a canvas of `width` x `height` pixels. */
export function drawFigure(
  ctx: CanvasRenderingContext2D,
  figure: Figure,
  box: Box,
  width: number,
  height: number,
  mirror: boolean,
) {
  const scale = Math.min(width / box.width, height / box.height);
  const flip = mirror ? -1 : 1;
  const place = (p: Point): Px => ({
    x: width / 2 + (p.x - box.cx) * scale * flip,
    y: height / 2 + (p.y - box.cy) * scale,
  });
  const pose = figure.pose.map(place);
  ctx.clearRect(0, 0, width, height);

  const [ls, rs] = [pose[11], pose[12]];
  const unit = distance(ls, rs) || 1; // shoulder width in pixels
  const rim = Math.max(1, unit * 0.025);
  const chest = between(ls, rs);

  // Torso: from the shoulders straight down past the bottom of the view. The
  // hips are out of the camera's shot in most takes, so they are not used.
  const across = { x: (ls.x - rs.x) / unit, y: (ls.y - rs.y) / unit };
  const down = across.x >= 0 ? { x: -across.y, y: across.x } : { x: across.y, y: -across.x };
  const waist = { x: chest.x + down.x * unit * 1.9, y: chest.y + down.y * unit * 1.9 };
  const side = (sign: number): Px => ({
    x: waist.x + across.x * unit * 0.36 * sign,
    y: waist.y + across.y * unit * 0.36 * sign,
  });
  shape(ctx, [ls, rs, side(-1), side(1)], unit * 0.22, BODY, rim, true);

  // Neck and head. The ears give the head's place and size, the eyes and nose its direction.
  const head = between(pose[L_EAR], pose[R_EAR]);
  const radius = Math.max(distance(pose[L_EAR], pose[R_EAR]) * 0.6, unit * 0.22);
  shape(ctx, [chest, head], unit * 0.24, BODY, rim);
  const [screenLeft, screenRight] =
    pose[L_EAR].x < pose[R_EAR].x ? [pose[L_EAR], pose[R_EAR]] : [pose[R_EAR], pose[L_EAR]];
  const tilt = Math.atan2(screenRight.y - screenLeft.y, screenRight.x - screenLeft.x);
  ctx.beginPath();
  ctx.ellipse(head.x, head.y, radius, radius * 1.3, tilt, 0, Math.PI * 2);
  ctx.fillStyle = BODY;
  ctx.fill();
  ctx.strokeStyle = EDGE;
  ctx.lineWidth = rim;
  ctx.stroke();
  for (const i of [L_EYE, R_EYE, NOSE]) dot(ctx, pose[i], radius * 0.09, EDGE);

  // Arms and hands, the farther arm first.
  const arms = [...ARMS].sort((a, b) => figure.pose[b.wrist].z - figure.pose[a.wrist].z);
  for (const arm of arms) {
    const hand = figure[arm.hand];
    const px = hand?.map(place);
    const elbow = pose[arm.elbow];
    const wrist = px ? px[0] : pose[arm.wrist];
    shape(ctx, [pose[arm.shoulder], elbow, wrist], unit * 0.19, BODY, rim);
    if (hand && px) {
      drawHand(ctx, hand, px, rim);
    } else {
      // The hand was not tracked here: a plain mitten on the end of the forearm.
      const reach = distance(elbow, wrist) || 1;
      const tip = {
        x: wrist.x + ((wrist.x - elbow.x) / reach) * unit * 0.1,
        y: wrist.y + ((wrist.y - elbow.y) / reach) * unit * 0.1,
      };
      dot(ctx, tip, unit * 0.14 + rim, EDGE);
      dot(ctx, tip, unit * 0.14, BODY);
    }
  }
}
