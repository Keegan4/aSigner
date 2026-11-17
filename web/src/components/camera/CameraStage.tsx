import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

/** What the camera view shows on top of the video. */
export type StageOverlay =
  | { kind: "none" }
  | { kind: "countdown"; left: number }
  /** A take in progress; the bar fills over `seconds`. */
  | { kind: "recording"; seconds: number }
  /** Waiting on something, for example "Checking…" or "Saving…". */
  | { kind: "busy"; label: string };

/**
 * The frame around a camera view (wrap CameraTracker in it), with the
 * countdown, recording and busy states drawn on the video where the learner is
 * looking. Overlays are purely visual: the page's own role="status" text
 * announces the same state, so they are hidden from assistive technology. They
 * are plain positioned elements with no blur, so tracking keeps the main thread.
 */
export function CameraStage({
  overlay = { kind: "none" },
  corner,
  className,
  children,
}: {
  overlay?: StageOverlay;
  /** A small status shown in the top-right corner, for example the connection. */
  corner?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  const recording = overlay.kind === "recording";
  return (
    <div
      className={cn(
        "relative rounded-card transition-shadow duration-150",
        recording && "ring-4 ring-danger ring-offset-2 ring-offset-background",
        className,
      )}
    >
      {children}
      <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden rounded-card">
        {overlay.kind === "countdown" && (
          <div className="absolute inset-0 flex items-center justify-center bg-stage/25">
            <span
              key={overlay.left}
              className="flex size-24 animate-pop items-center justify-center rounded-full bg-stage/75 text-5xl font-semibold text-on-stage tabular-nums ring-2 ring-on-stage/40 sm:size-32 sm:text-7xl"
            >
              {overlay.left}
            </span>
          </div>
        )}
        {recording && (
          <>
            <span className="absolute top-3 left-3 inline-flex items-center gap-2 rounded-full bg-danger px-3 py-1 text-xs font-semibold tracking-wider text-on-danger uppercase shadow-card">
              <span className="size-2 rounded-full bg-on-danger" />
              Recording
            </span>
            <span className="absolute inset-x-0 bottom-0 h-2 bg-stage/60">
              <span
                className="block h-full origin-left animate-record-bar bg-danger"
                style={{ animationDuration: `${overlay.seconds}s` }}
              />
            </span>
          </>
        )}
        {overlay.kind === "busy" && (
          <div className="absolute inset-0 flex items-center justify-center bg-stage/25">
            <span className="inline-flex items-center gap-2.5 rounded-full bg-stage/80 px-4 py-2 text-sm font-medium text-on-stage shadow-card">
              <span className="spinner size-4" />
              {overlay.label}
            </span>
          </div>
        )}
        {corner && <div className="absolute top-3 right-3">{corner}</div>}
      </div>
    </div>
  );
}

/** A dark pill for CameraStage's corner. */
export function StagePill({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-stage/80 px-3 py-1 text-xs font-medium text-on-stage">
      {children}
    </span>
  );
}
