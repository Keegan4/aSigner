"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { IconEye, IconEyeOff } from "@/components/ui/Icons";
import { Meter } from "@/components/ui/Meter";
import { SectionHeading } from "@/components/ui/SectionHeading";
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


/** The model's top guesses for the latest window, each with its softmax probability. */
export function PredictionsPanel({ prediction }: { prediction: Prediction | null }) {
  return (
    <Card as="section" aria-label="Top predictions" className="flex flex-col gap-3 text-sm">
      <SectionHeading variant="eyebrow">Top 5 predictions</SectionHeading>
      {!prediction ? (
        <p className="text-muted">Waiting for the first window…</p>
      ) : prediction.top.length === 0 ? (
        <p className="text-muted">No prediction: too little seen yet, or no hands in view.</p>
      ) : (
        <ol className="flex flex-col gap-2.5">
          {prediction.top.map(([gloss, p], i) => {
            const name = gloss === IDLE_LABEL ? "(no sign)" : gloss;
            return (
              <li key={gloss} className="flex flex-col gap-1">
                <div className="flex items-baseline justify-between gap-2 font-mono">
                  <span className={i === 0 ? "font-semibold text-foreground" : "text-foreground"}>
                    {i + 1}. {name}
                  </span>
                  <span className="text-muted tabular-nums">{percent(p)}</span>
                </div>
                <Meter
                  size="sm"
                  value={p}
                  label={`${name} probability`}
                  valueText={percent(p)}
                  tone={i === 0 ? "success" : "neutral"}
                />
              </li>
            );
          })}
        </ol>
      )}
      {prediction && (
        <p className="border-t border-line pt-3 text-muted">
          Read as: <span className="font-mono font-medium text-foreground">{prediction.label ?? "no sign"}</span>
          <span className="block text-xs">Before the check that a sign is held for several windows.</span>
        </p>
      )}
    </Card>
  );
}

const fmt = (v: number) => v.toFixed(3).padStart(7);

function Points({ frame, offset, count }: { frame: Float32Array; offset: number; count: number }) {
  const rows = [];
  for (let i = 0; i < count; i++) {
    const [x, y, z] = [frame[offset + i * 3], frame[offset + i * 3 + 1], frame[offset + i * 3 + 2]];
    rows.push(`${String(i).padStart(2)} ${fmt(x)} ${fmt(y)} ${fmt(z)}`);
  }
  return (
    <pre className="overflow-x-auto rounded-md bg-sunken p-2 font-mono text-xs leading-snug">
      {"    x       y       z\n" + rows.join("\n")}
    </pre>
  );
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
    <details open={present} className="group rounded-control border border-line px-3 py-2">
      <summary className="flex min-h-7 cursor-pointer items-center justify-between gap-2 font-medium">
        {title}{" "}
        <Badge
          tone={present ? "success" : "neutral"}
          icon={present ? <IconEye size={14} /> : <IconEyeOff size={14} />}
        >
          {present ? "seen" : "not seen"}
        </Badge>
      </summary>
      {present && <div className="mt-2">{children}</div>}
    </details>
  );
}

/** Written by the page as frames arrive; read by LandmarksPanel a few times a second. */
export type FrameFeed = {
  /** The newest landmark frame. */
  frame: Float32Array | null;
  /** Landmark frames produced by tracking so far. */
  sent: number;
  /** Landmark points (hand and pose) found in those frames so far. */
  points: number;
  /** Camera frames shown so far, whether or not tracking kept up with them. */
  videoFrames: number;
  /** The camera's nominal frame rate, if it reports one. */
  cameraFps?: number;
};

export const emptyFeed = (): FrameFeed => ({ frame: null, sent: 0, points: 0, videoFrames: 0 });

/** How many hand and pose landmarks a packed frame holds (missing ones are all zeros). */
export function landmarkCount(frame: Float32Array): number {
  let count = 0;
  for (let i = LEFT_HAND; i < POSE + POSE_POINTS * 3; i += 3) {
    if (frame[i] !== 0 || frame[i + 1] !== 0 || frame[i + 2] !== 0) count++;
  }
  return count;
}

type Rates = { video: number; tracked: number; points: number };

/**
 * The raw landmark values of the frames being sent (the packed layout of
 * lib/tracking/features.ts), refreshed a few times a second.
 */
export function LandmarksPanel({ feed }: { feed: React.RefObject<FrameFeed> }) {
  const [frame, setFrame] = useState<Float32Array | null>(null);
  const [rates, setRates] = useState<Rates | null>(null);

  useEffect(() => {
    let since = performance.now();
    let before = { ...feed.current };
    const timer = setInterval(() => {
      const current = feed.current;
      setFrame(current.frame);
      const now = performance.now();
      if (now - since >= 1000) {
        const perSecond = (n: number, m: number) => ((n - m) * 1000) / (now - since);
        setRates({
          video: perSecond(current.videoFrames, before.videoFrames),
          tracked: perSecond(current.sent, before.sent),
          points: perSecond(current.points, before.points),
        });
        before = { ...current };
        since = now;
      }
    }, 200);
    return () => clearInterval(timer);
  }, [feed]);

  const deg = (r: number) => `${((r * 180) / Math.PI).toFixed(1)}°`;

  return (
    <Card as="section" aria-label="Landmark output" className="flex flex-col gap-3 text-sm">
      <SectionHeading variant="eyebrow">Landmark output</SectionHeading>
      {!frame ? (
        <p className="text-muted">Waiting for the camera…</p>
      ) : (
        <>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-xs">
            <dt className="text-muted">Video</dt>
            <dd>
              {rates && feed.current.videoFrames > 0 ? `${rates.video.toFixed(1)} fps` : "–"}
              {feed.current.cameraFps ? (
                <span className="text-muted"> (camera {feed.current.cameraFps.toFixed(0)})</span>
              ) : null}
            </dd>
            <dt className="text-muted">Tracked</dt>
            <dd>{rates ? `${rates.tracked.toFixed(1)} frames/s` : "–"}</dd>
            <dt className="text-muted">Landmarks</dt>
            <dd>{rates ? `${rates.points.toFixed(0)} points/s` : "–"}</dd>
            <dt className="text-muted">This frame</dt>
            <dd>{landmarkCount(frame)} points</dd>
          </dl>
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
            <pre className="overflow-x-auto rounded-md bg-sunken p-2 font-mono text-xs leading-snug">
              {Array.from(frame.subarray(FACE, FACE + BLENDSHAPES))
                .map((v, i) => `${String(i).padStart(2)} ${v.toFixed(3)}`)
                .join("\n")}
            </pre>
          </Part>
        </>
      )}
    </Card>
  );
}
