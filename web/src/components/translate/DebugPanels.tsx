"use client";

import { useEffect, useState } from "react";

import {
  BLENDSHAPES,
  FACE,
  HAND_POINTS,
  HEAD,
  LEFT_HAND,
  POSE,
  POSE_POINTS,
  RIGHT_HAND,
} from "@/lib/tracking/features";

/** One model prediction from /ws/translate: the sign it was read as and the top guesses. */
export type Prediction = { label: string | null; top: [string, number][] };

const IDLE_LABEL = "_idle"; // server/app/recognition/features.py
const percent = (p: number) => `${(p * 100).toFixed(1)}%`;

const panel = "flex flex-col gap-3 rounded-lg border p-4 text-sm";
const heading = "text-sm font-medium uppercase tracking-wide opacity-60";

/** The model's top guesses for the latest window, each with its softmax probability. */
export function PredictionsPanel({ prediction }: { prediction: Prediction | null }) {
  return (
    <section className={panel} aria-label="Top predictions">
      <h2 className={heading}>Top 5 predictions</h2>
      {!prediction ? (
        <p className="opacity-60">Waiting for the first window…</p>
      ) : prediction.top.length === 0 ? (
        <p className="opacity-60">No prediction: too little seen yet, or no hands in view.</p>
      ) : (
        <ol className="flex flex-col gap-2">
          {prediction.top.map(([gloss, p], i) => (
            <li key={gloss} className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between gap-2 font-mono">
                <span className={i === 0 ? "font-semibold" : ""}>
                  {i + 1}. {gloss === IDLE_LABEL ? "(no sign)" : gloss}
                </span>
                <span className="tabular-nums opacity-80">{percent(p)}</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-foreground/10">
                <div
                  className={`h-full rounded-full ${i === 0 ? "bg-emerald-500" : "bg-foreground/40"}`}
                  style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ol>
      )}
      {prediction && (
        <p className="opacity-70">
          Read as: <span className="font-mono">{prediction.label ?? "no sign"}</span>
          <span className="block text-xs opacity-80">
            Before the check that a sign is held for several windows.
          </span>
        </p>
      )}
    </section>
  );
}

const fmt = (v: number) => v.toFixed(3).padStart(7);

function Points({ frame, offset, count }: { frame: Float32Array; offset: number; count: number }) {
  const rows = [];
  for (let i = 0; i < count; i++) {
    const [x, y, z] = [frame[offset + i * 3], frame[offset + i * 3 + 1], frame[offset + i * 3 + 2]];
    rows.push(`${String(i).padStart(2)} ${fmt(x)} ${fmt(y)} ${fmt(z)}`);
  }
  return <pre className="overflow-x-auto text-xs leading-snug">{"    x       y       z\n" + rows.join("\n")}</pre>;
}

const seen = (frame: Float32Array, offset: number, size: number) =>
  frame.subarray(offset, offset + size).some((v) => v !== 0);

function Part({
  title,
  present,
  children,
}: {
  title: string;
  present: boolean;
  children: React.ReactNode;
}) {
  return (
    <details open={present} className="rounded-md border px-2 py-1">
      <summary className="cursor-pointer font-medium">
        {title}{" "}
        <span className={present ? "text-emerald-600" : "opacity-50"}>{present ? "seen" : "not seen"}</span>
      </summary>
      {present && <div className="mt-1">{children}</div>}
    </details>
  );
}

/** The newest frame sent and how many have been sent; written by the page on every frame. */
export type FrameFeed = { frame: Float32Array | null; sent: number };

/**
 * The raw landmark values of the frames being sent (the packed layout of
 * lib/tracking/features.ts), refreshed a few times a second.
 */
export function LandmarksPanel({ feed }: { feed: React.RefObject<FrameFeed> }) {
  const [frame, setFrame] = useState<Float32Array | null>(null);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let since = performance.now();
    let sentBefore = feed.current.sent;
    const timer = setInterval(() => {
      setFrame(feed.current.frame);
      const now = performance.now();
      if (now - since >= 1000) {
        setFps(((feed.current.sent - sentBefore) * 1000) / (now - since));
        sentBefore = feed.current.sent;
        since = now;
      }
    }, 200);
    return () => clearInterval(timer);
  }, [feed]);

  const deg = (r: number) => `${((r * 180) / Math.PI).toFixed(1)}°`;

  return (
    <section className={panel} aria-label="Landmark output">
      <h2 className={heading}>Landmark output</h2>
      {!frame ? (
        <p className="opacity-60">Waiting for the camera…</p>
      ) : (
        <>
          <p className="font-mono text-xs opacity-70">{fps.toFixed(0)} frames/s sent</p>
          <Part title="Head" present={seen(frame, HEAD, 3)}>
            <p className="font-mono text-xs">
              yaw {deg(frame[HEAD])} · pitch {deg(frame[HEAD + 1])} · roll {deg(frame[HEAD + 2])}
            </p>
          </Part>
          <Part title="Left hand" present={seen(frame, LEFT_HAND, HAND_POINTS * 3)}>
            <Points frame={frame} offset={LEFT_HAND} count={HAND_POINTS} />
          </Part>
          <Part title="Right hand" present={seen(frame, RIGHT_HAND, HAND_POINTS * 3)}>
            <Points frame={frame} offset={RIGHT_HAND} count={HAND_POINTS} />
          </Part>
          <Part title="Pose" present={seen(frame, POSE, POSE_POINTS * 3)}>
            <Points frame={frame} offset={POSE} count={POSE_POINTS} />
          </Part>
          <Part title="Face blendshapes" present={seen(frame, FACE, BLENDSHAPES)}>
            <pre className="overflow-x-auto text-xs leading-snug">
              {Array.from(frame.subarray(FACE, FACE + BLENDSHAPES))
                .map((v, i) => `${String(i).padStart(2)} ${v.toFixed(3)}`)
                .join("\n")}
            </pre>
          </Part>
        </>
      )}
    </section>
  );
}
