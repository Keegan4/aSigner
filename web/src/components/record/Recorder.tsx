"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraStage, type StageOverlay } from "@/components/camera/CameraStage";
import { CameraTracker } from "@/components/camera/CameraTracker";
import { CaptureControls, CaptureSign, SignChips } from "@/components/camera/CapturePanel";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Shell } from "@/components/ui/Shell";
import { LoadingBlock, ServerUnreachable, SignInPrompt } from "@/components/ui/States";
import { getJson, type LessonSummary, type Me } from "@/lib/api";
import { CLIP_SECONDS, FRAME_SIZE, LEFT_HAND, POSE } from "@/lib/tracking/features";

type SignRow = { id: number; gloss: string };
type Phase = "idle" | "countdown" | "recording" | "saving";
type Saved = { id: string; label: string };

const COUNTDOWN_SECONDS = 3;
// A sign's take is kept when a hand was tracked in at least this share of its
// frames. Low, because tracking drops out while a hand moves fast.
const MIN_HAND_SHARE = 0.25;
const TARGET_PER_SIGN = 25;
// Must match IDLE_LABEL in server/app/recognition/features.py.
const IDLE: SignRow = { id: -1, gloss: "_idle" };
// The traced letters, and every word: only a single letter can be a held shape.
const moves = (gloss: string) => gloss.length > 1 || gloss === "J" || gloss === "Z";

const hasHand = (frame: Float32Array) => frame.subarray(LEFT_HAND, POSE).some((v) => v !== 0);

/** Records short labelled landmark clips for training. Nothing but landmarks is uploaded. */
export function Recorder() {
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const [signs, setSigns] = useState<SignRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [last, setLast] = useState<Saved | null>(null);

  const phaseRef = useRef<Phase>("idle");
  const buffer = useRef<Float32Array[]>([]);
  const startedAt = useRef(0);

  const sign = signs[index];

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    const everySign = async () => {
      const lessons = (await getJson<LessonSummary[]>("/api/lessons")) ?? [];
      const loaded = await Promise.all(
        lessons.map((l) => getJson<{ signs: SignRow[] }>(`/api/lessons/${encodeURIComponent(l.slug)}`)),
      );
      return loaded.flatMap((lesson) => lesson?.signs ?? []);
    };
    Promise.all([
      getJson<Me>("/api/me"),
      everySign(),
      getJson<Record<string, number>>("/api/clips/counts"),
    ])
      .then(([user, all, saved]) => {
        setMe(user);
        setSigns([...all, IDLE]);
        setCounts(saved ?? {});
      })
      .catch(() => {
        setMe(null);
        setMessage("The aSigner server is not reachable.");
      });
  }, []);

  const onFrame = useCallback((frame: Float32Array) => {
    if (phaseRef.current === "recording") buffer.current.push(frame);
  }, []);

  const upload = useCallback(async (label: string) => {
    const frames = buffer.current;
    const elapsed = (performance.now() - startedAt.current) / 1000;
    if (frames.length < 5) {
      setMessage("Too few frames were captured. Check the camera and try again.");
      return;
    }
    // A resting take may have no hands in view at all, so the check is for signs only.
    if (label !== IDLE.gloss && frames.filter(hasHand).length < frames.length * MIN_HAND_SHARE) {
      setMessage("Hardly any of that take showed a hand, so it was not saved.");
      return;
    }
    const body = new Float32Array(frames.length * FRAME_SIZE);
    frames.forEach((frame, i) => body.set(frame, i * FRAME_SIZE));
    const query = new URLSearchParams({ label, fps: (frames.length / elapsed).toFixed(2) });
    const res = await fetch(`/api/clips?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
    });
    if (!res.ok) {
      setMessage(`The clip was not saved (server answered ${res.status}).`);
      return;
    }
    const saved: Saved = await res.json();
    setLast(saved);
    setCounts((c) => ({ ...c, [label]: (c[label] ?? 0) + 1 }));
    setMessage(null);
  }, []);

  const record = useCallback(() => {
    if (!sign || phaseRef.current !== "idle") return;
    const label = sign.gloss;
    setMessage(null);
    setPhase("countdown");
    setSeconds(COUNTDOWN_SECONDS);
    let left = COUNTDOWN_SECONDS;
    const timer = setInterval(() => {
      left -= 1;
      setSeconds(left);
      if (left > 0) return;
      clearInterval(timer);
      buffer.current = [];
      startedAt.current = performance.now();
      setPhase("recording");
      phaseRef.current = "recording";
      setTimeout(async () => {
        setPhase("saving");
        phaseRef.current = "saving";
        await upload(label);
        setPhase("idle");
      }, CLIP_SECONDS * 1000);
    }, 1000);
  }, [sign, upload]);

  const undo = useCallback(async () => {
    if (!last) return;
    const res = await fetch(`/api/clips/${last.label}/${last.id}`, { method: "DELETE" });
    if (res.ok) {
      setCounts((c) => ({ ...c, [last.label]: Math.max(0, (c[last.label] ?? 1) - 1) }));
      setLast(null);
    }
  }, [last]);

  const step = useCallback(
    (by: number) => setIndex((i) => (signs.length ? (i + by + signs.length) % signs.length : 0)),
    [signs.length],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Keys aimed at a control (the header's menu, a button, a field) are left to it.
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, textarea, button, select")) return;
      if (e.code === "Space") {
        e.preventDefault();
        record();
      } else if (e.code === "ArrowRight") step(1);
      else if (e.code === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [record, step]);

  if (me === undefined) {
    return (
      <Frame>
        <LoadingBlock lines={4} />
      </Frame>
    );
  }
  if (me === null) {
    return (
      <Frame>
        {message ? <ServerUnreachable message={message} /> : <SignInPrompt to="record training clips" />}
      </Frame>
    );
  }

  const done = sign ? (counts[sign.gloss] ?? 0) : 0;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const idle = sign?.gloss === IDLE.gloss;
  const name = (s: SignRow) => (s.gloss === IDLE.gloss ? "No sign" : s.gloss);
  const how = idle
    ? "Keep your hands in view and move them as you would between signs, without making one. Record a few takes with your hands resting too."
    : sign && moves(sign.gloss)
      ? "Start from the first handshape and make the whole movement during the take."
      : "Have the sign ready before the countdown ends and keep it up for the whole take.";
  const prompt = {
    idle: `Press Space or Record. ${how}`,
    countdown: `Get the sign ready… ${seconds}`,
    recording: "Recording: keep signing.",
    saving: "Saving…",
  }[phase];
  const overlay: StageOverlay =
    phase === "countdown"
      ? { kind: "countdown", left: seconds }
      : phase === "recording"
        ? { kind: "recording", seconds: CLIP_SECONDS }
        : phase === "saving"
          ? { kind: "busy", label: "Saving…" }
          : { kind: "none" };

  return (
    <Frame>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:grid-rows-[auto_1fr]">
        <div className="lg:col-start-2 lg:row-start-1">
          <CaptureSign
            label={sign ? name(sign) : "–"}
            count={done}
            target={TARGET_PER_SIGN}
            countText={`${done} of ${TARGET_PER_SIGN} for this sign`}
            details={`${total} clips in total`}
          />
        </div>

        <div className="flex min-w-0 flex-col gap-4 lg:col-start-1 lg:row-span-2 lg:row-start-1">
          <CameraStage overlay={overlay}>
            <CameraTracker onFrame={onFrame} />
          </CameraStage>

          <p role="status" className="text-lg font-medium text-foreground">
            {prompt}
          </p>
          {message && (
            <Callout tone="danger" role="alert">
              {message}
            </Callout>
          )}

          <CaptureControls
            onPrevious={() => step(-1)}
            onRecord={record}
            onNext={() => step(1)}
            onUndo={undo}
            canRecord={phase === "idle" && Boolean(sign)}
            canUndo={Boolean(last) && phase === "idle"}
          />

          <p className="text-sm text-muted">
            Each take is {CLIP_SECONDS === 1 ? "one second" : `${CLIP_SECONDS} seconds`}. Between takes, change your distance, position and hand
            angle a little. Only landmark coordinates are saved; no video leaves this tab.
          </p>
        </div>

        <Card as="section" aria-labelledby="record-signs" className="flex flex-col gap-3 lg:col-start-2 lg:row-start-2 lg:self-start">
          <SectionHeading id="record-signs" variant="eyebrow">
            Signs
          </SectionHeading>
          <SignChips
            items={signs.map((s, i) => {
              const n = counts[s.gloss] ?? 0;
              return {
                key: String(i),
                label: name(s),
                count: n,
                current: i === index,
                done: n >= TARGET_PER_SIGN,
              };
            })}
            onSelect={(key) => setIndex(Number(key))}
          />
        </Card>
      </div>
    </Frame>
  );
}

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <Shell back="/" backLabel="Lessons" title="Record training clips" width="default">
      {children}
    </Shell>
  );
}
