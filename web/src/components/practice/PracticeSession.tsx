"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraStage, type StageOverlay } from "@/components/camera/CameraStage";
import { CameraTracker } from "@/components/camera/CameraTracker";
import { Button } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { IconAlert, IconCircleCheck, IconRecord } from "@/components/ui/Icons";
import { Meter } from "@/components/ui/Meter";
import { ModelNote } from "@/components/ui/ModelNote";
import { ServerUnreachable, SignInPrompt } from "@/components/ui/States";
import type { ModelInfo, Sign } from "@/lib/api";
import { CLIP_SECONDS } from "@/lib/tracking/features";
import {
  MOCK_PRACTICE,
  openPractice,
  type PracticeEvent,
  type PracticeLink,
  type PracticeResult,
} from "@/lib/practice";
import { cn } from "@/lib/cn";

const COUNTDOWN = 3;
// Same countdown and take length as the training recorders (Recorder.tsx,
// Trainer.tsx), so practice clips look like the clips the model was trained on.
const RECORD_MS = CLIP_SECONDS * 1000;

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
          if (["no_model", "bad_model", "unsupported_sign"].includes(event.code)) {
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

  const overlay: StageOverlay =
    phase.name === "countdown"
      ? { kind: "countdown", left: phase.left }
      : phase.name === "recording"
        ? { kind: "recording", seconds: CLIP_SECONDS }
        : phase.name === "starting"
          ? { kind: "busy", label: "Starting…" }
          : phase.name === "scoring"
            ? { kind: "busy", label: "Checking…" }
            : { kind: "none" };

  return (
    <div className="flex flex-col gap-4">
      {MOCK_PRACTICE && (
        <Callout tone="warning" title="Demo mode">
          Demo mode: results are made up and nothing is saved.
        </Callout>
      )}
      <CameraStage overlay={overlay}>
        <CameraTracker onFrame={onFrame} />
      </CameraStage>

      {status === "signed-out" ? (
        <SignInPrompt role="alert" to="practise and keep your progress" />
      ) : status === "closed" ? (
        <ServerUnreachable reason="lost" />
      ) : (
        <>
          <div className="flex flex-col gap-3 rounded-card border border-line bg-surface p-3 shadow-card sm:flex-row sm:items-center sm:gap-4">
            <Button
              variant="primary"
              size="lg"
              icon={<IconRecord size={20} />}
              onClick={record}
              disabled={busy || status !== "open"}
              className="sm:min-w-40"
            >
              {phase.name === "result" ? "Try again" : "Record"}
            </Button>
            <p role="status" aria-live="polite" className="text-sm font-medium text-foreground">
              {status === "connecting" ? "Connecting…" : prompt}
            </p>
          </div>
          {unavailable && (
            <Callout tone="warning" role="alert">
              {unavailable}
            </Callout>
          )}
          {phase.name === "idle" && phase.error && (
            <Callout tone="danger" role="alert">
              {phase.error}
            </Callout>
          )}
        </>
      )}

      {phase.name === "result" && (
        <ResultBanner result={phase.result} gloss={sign.gloss}>
          {children}
        </ResultBanner>
      )}

      <ModelNote model={model} />
    </div>
  );
}

/** The outcome of one attempt: an icon and words as well as colour, and the score as a bar. */
function ResultBanner({
  result,
  gloss,
  children,
}: {
  result: PracticeResult;
  gloss: string;
  children?: React.ReactNode;
}) {
  const percent = Math.round(result.score * 100);
  return (
    <div
      role="status"
      className={cn(
        "flex flex-col gap-3 rounded-card border p-4 sm:p-5",
        result.passed ? "border-success-line bg-success-soft" : "border-warning-line bg-warning-soft",
      )}
    >
      <div className="flex items-start gap-3">
        <span className={result.passed ? "text-success-fg" : "text-warning-fg"}>
          {result.passed ? <IconCircleCheck size={28} /> : <IconAlert size={28} />}
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <p className="text-lg font-semibold text-foreground">
            {result.passed ? "Recognised" : "Not recognised yet"}
            <span className="sr-only"> · </span>
            <span className="block text-sm font-medium text-foreground">
              {percent}% match for {gloss}
            </span>
          </p>
          <Meter
            value={result.score}
            label={`Match for ${gloss}`}
            valueText={`${percent}%`}
            tone={result.passed ? "success" : "warning"}
          />
          {result.closest && <p className="text-sm text-foreground">It looked more like {result.closest}.</p>}
          {result.no_sign && (
            <p className="text-sm text-foreground">No clear sign was seen. Check that your hand is in view.</p>
          )}
        </div>
      </div>
      {children && <div className="flex flex-wrap gap-2 sm:pl-10">{children}</div>}
    </div>
  );
}
