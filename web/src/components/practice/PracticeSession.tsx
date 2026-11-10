"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraTracker } from "@/components/camera/CameraTracker";
import { ModelNote } from "@/components/ui/ModelNote";
import type { ModelInfo, Sign } from "@/lib/api";
import {
  MOCK_PRACTICE,
  openPractice,
  type PracticeEvent,
  type PracticeLink,
  type PracticeResult,
} from "@/lib/practice";

const COUNTDOWN = 3;
// Same countdown and take length as the training recorder (Recorder.tsx), so
// practice clips look like the clips the model was trained on.
const RECORD_MS = 2000;

type Link_ = "connecting" | "open" | "signed-out" | "closed";
type Phase =
  | { name: "idle"; error?: string }
  | { name: "countdown"; left: number }
  | { name: "starting" }
  | { name: "recording" }
  | { name: "scoring" }
  | { name: "result"; result: PracticeResult };

/**
 * Camera plus one scored attempt at a time for `sign`. The parent decides what
 * comes next through `onResult` and by changing `sign`.
 */
export function PracticeSession({
  sign,
  onResult,
  children,
}: {
  sign: Sign;
  onResult?: (result: PracticeResult) => void;
  /** Shown under the result, for example a Next button. */
  children?: React.ReactNode;
}) {
  const link = useRef<PracticeLink | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const recording = useRef(false);
  const onResultRef = useRef(onResult);
  const [status, setStatus] = useState<Link_>("connecting");
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  // Tagged with the sign so a new sign starts idle without an effect.
  const [tagged, setTagged] = useState<{ signId: number; phase: Phase }>({
    signId: sign.id,
    phase: { name: "idle" },
  });
  const phase: Phase = tagged.signId === sign.id ? tagged.phase : { name: "idle" };
  const signId = sign.id;
  const setPhase = useCallback((next: Phase) => setTagged({ signId, phase: next }), [signId]);

  useEffect(() => {
    onResultRef.current = onResult;
  }, [onResult]);

  const handlers = useRef<(event: PracticeEvent) => void>(() => {});
  useEffect(() => {
    handlers.current = (event) => {
      switch (event.type) {
        case "open":
          setStatus("open");
          break;
        case "closed":
          recording.current = false;
          clearTimeout(timer.current);
          setStatus(event.signedOut ? "signed-out" : "closed");
          setPhase({ name: "idle" });
          break;
        case "ready":
          setModel(event.model);
          recording.current = true;
          setPhase({ name: "recording" });
          timer.current = setTimeout(() => {
            recording.current = false;
            link.current?.score();
            setPhase({ name: "scoring" });
          }, RECORD_MS);
          break;
        case "result":
          setPhase({ name: "result", result: event });
          onResultRef.current?.(event);
          break;
        case "error":
          recording.current = false;
          clearTimeout(timer.current);
          if (event.code === "no_model" || event.code === "unsupported_sign") {
            setUnavailable(event.message);
            setPhase({ name: "idle" });
          } else if (event.code !== "bad_frame") {
            setPhase({ name: "idle", error: event.message });
          }
          break;
      }
    };
  }, [setPhase]);

  useEffect(() => {
    const opened = openPractice((event) => handlers.current(event));
    link.current = opened;
    return () => {
      clearTimeout(timer.current);
      opened.close();
    };
  }, []);

  // A new sign abandons whatever attempt was in flight.
  useEffect(() => {
    return () => {
      clearTimeout(timer.current);
      recording.current = false;
    };
  }, [signId]);

  const onFrame = useCallback((frame: Float32Array) => {
    if (recording.current) link.current?.frame(frame);
  }, []);

  function record() {
    setUnavailable(null);
    const tick = (left: number) => {
      if (left === 0) {
        setPhase({ name: "starting" });
        link.current?.start(signId);
        return;
      }
      setPhase({ name: "countdown", left });
      timer.current = setTimeout(() => tick(left - 1), 1000);
    };
    tick(COUNTDOWN);
  }

  const busy = phase.name !== "idle" && phase.name !== "result";
  const prompt = {
    idle: "Press Record and have the sign ready before the countdown ends.",
    countdown: `Get the sign ready… ${phase.name === "countdown" ? phase.left : ""}`,
    starting: "Starting…",
    recording: "Recording: keep signing.",
    scoring: "Checking…",
    result: "",
  }[phase.name];

  return (
    <div className="flex flex-col gap-4">
      {MOCK_PRACTICE && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
          Demo mode: results are made up and nothing is saved.
        </p>
      )}
      <CameraTracker onFrame={onFrame} />

      {status === "signed-out" ? (
        <p role="alert" className="text-sm">
          <a href="/api/auth/login" className="underline underline-offset-4">
            Sign in with Google
          </a>{" "}
          to practise and keep your progress.
        </p>
      ) : status === "closed" ? (
        <p role="alert" className="text-sm text-red-600">
          The connection to the aSigner server was lost. Reload to try again.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={record}
              disabled={busy || status !== "open"}
              className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background disabled:opacity-40"
            >
              {phase.name === "result" ? "Try again" : "Record"}
            </button>
            <p role="status" aria-live="polite" className="text-sm">
              {status === "connecting" ? "Connecting…" : prompt}
            </p>
          </div>
          {unavailable && (
            <p role="alert" className="rounded-md border p-3 text-sm">
              {unavailable}
            </p>
          )}
          {phase.name === "idle" && phase.error && (
            <p role="alert" className="text-sm text-red-600">
              {phase.error}
            </p>
          )}
        </>
      )}

      <ModelNote model={model} />

      {phase.name === "result" && (
        <div
          role="status"
          className={`flex flex-col gap-2 rounded-lg border p-4 ${
            phase.result.passed ? "border-emerald-500/50 bg-emerald-500/10" : "border-amber-500/50 bg-amber-500/10"
          }`}
        >
          <p className="font-medium">
            {phase.result.passed ? "Recognised" : "Not recognised yet"} ·{" "}
            {Math.round(phase.result.score * 100)}% match for {sign.gloss}
          </p>
          {phase.result.closest && (
            <p className="text-sm">It looked more like {phase.result.closest}.</p>
          )}
          {phase.result.no_sign && (
            <p className="text-sm">No clear sign was seen. Check that your hand is in view.</p>
          )}
          {children}
        </div>
      )}
    </div>
  );
}
