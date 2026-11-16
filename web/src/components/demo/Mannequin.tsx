"use client";

import { useEffect, useRef, useState } from "react";
import { type DemoClip, duration, figureAt, frameBox, loadDemo } from "@/lib/demo/clip";
import { drawFigure } from "@/lib/demo/draw";

/** How long the figure holds the last frame before the clip starts again. */
const HOLD_SECONDS = 0.6;

function prefersStillness() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * A drawn figure that plays back one recorded take of a sign, from the
 * landmark clip at `src` (see ml/export_demos.py). It faces the learner as a
 * teacher would; "Mirror" flips it for learners who would rather copy a mirror.
 */
export function Mannequin({ src, gloss }: { src: string; gloss: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [clip, setClip] = useState<DemoClip | null>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(() => !prefersStillness());
  const [slow, setSlow] = useState(false);
  const [mirror, setMirror] = useState(false);
  const settings = useRef({ playing, slow, mirror });
  /** Redraws, and restarts the loop if it should be running. */
  const wake = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadDemo(src).then(
      (loaded) => {
        if (cancelled) return;
        setFailed(false);
        setClip(loaded);
      },
      () => {
        if (!cancelled) setFailed(true);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [src]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!clip || !canvas || !ctx) return;
    const box = frameBox(clip);
    const length = duration(clip);
    // Paused from the start (reduced motion): show the middle of the sign, not its wind-up.
    let at = settings.current.playing ? 0 : length / 2;
    let raf = 0;
    let last = 0;
    let visible = false;

    const tick = (now: number) => {
      raf = 0;
      const { playing, slow, mirror } = settings.current;
      if (playing && last) {
        // A tab left in the background resumes with a huge gap: do not skip ahead by it.
        at += (Math.min(now - last, 100) / 1000) * (slow ? 0.5 : 1);
        if (at > length + HOLD_SECONDS) at = 0;
      }
      last = now;
      const ratio = window.devicePixelRatio || 1;
      const width = Math.round(canvas.clientWidth * ratio);
      const height = Math.round(canvas.clientHeight * ratio);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      if (width && height) drawFigure(ctx, figureAt(clip, at), box, width, height, mirror);
      if (playing && visible) raf = requestAnimationFrame(tick);
      else last = 0;
    };
    const run = () => {
      if (!raf) raf = requestAnimationFrame(tick);
    };
    wake.current = run;

    // A lesson page shows many figures at once: only the ones on screen animate.
    const watcher = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) run();
    });
    watcher.observe(canvas);
    const resizer = new ResizeObserver(run);
    resizer.observe(canvas);
    return () => {
      wake.current = null;
      cancelAnimationFrame(raf);
      watcher.disconnect();
      resizer.disconnect();
    };
  }, [clip]);

  useEffect(() => {
    settings.current = { playing, slow, mirror };
    wake.current?.();
  }, [playing, slow, mirror]);

  if (failed) {
    return (
      <div
        role="alert"
        className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed text-sm opacity-60"
      >
        The demonstration could not be loaded
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Demonstration of ${gloss}`}
        className="aspect-video w-full rounded-lg bg-neutral-900"
      />
      <div className="flex gap-1.5 text-xs">
        <Toggle on={playing} onChange={setPlaying} label={playing ? "Pause" : "Play"} />
        <Toggle on={slow} onChange={setSlow} label="Slow" />
        <Toggle on={mirror} onChange={setMirror} label="Mirror" />
      </div>
    </div>
  );
}

function Toggle({ on, onChange, label }: { on: boolean; onChange: (on: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => onChange(!on)}
      className={`rounded border px-2 py-0.5 ${on ? "bg-foreground text-background" : ""}`}
    >
      {label}
    </button>
  );
}
