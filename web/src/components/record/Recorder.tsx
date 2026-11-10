"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CameraTracker } from "@/components/camera/CameraTracker";
import { getJson, type Me } from "@/lib/api";
import { FRAME_SIZE, LEFT_HAND, POSE } from "@/lib/tracking/features";

type SignRow = { id: number; gloss: string };
type Phase = "idle" | "countdown" | "recording" | "saving";
type Saved = { id: string; label: string };

const COUNTDOWN_SECONDS = 3;
const CLIP_SECONDS = 2;
const TARGET_PER_SIGN = 25;
// Must match IDLE_LABEL in server/app/recognition/features.py.
const IDLE: SignRow = { id: -1, gloss: "_idle" };
const MOVING = new Set(["J", "Z"]);

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
    Promise.all([
      getJson<Me>("/api/me"),
      getJson<{ signs: SignRow[] }>("/api/lessons/alphabet"),
      getJson<Record<string, number>>("/api/clips/counts"),
    ])
      .then(([user, lesson, saved]) => {
        setMe(user);
        setSigns([...(lesson?.signs ?? []), IDLE]);
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
    if (label !== IDLE.gloss && frames.filter(hasHand).length < frames.length / 2) {
      setMessage("No hand was visible for most of that take, so it was not saved.");
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
      if (e.code === "Space") {
        e.preventDefault();
        record();
      } else if (e.code === "ArrowRight") step(1);
      else if (e.code === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [record, step]);

  if (me === undefined) return <Shell>Loading…</Shell>;
  if (me === null) {
    return <Shell>{message ?? "Sign in on the home page to record training clips."}</Shell>;
  }

  const done = sign ? (counts[sign.gloss] ?? 0) : 0;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const idle = sign?.gloss === IDLE.gloss;
  const name = (s: SignRow) => (s.gloss === IDLE.gloss ? "No sign" : s.gloss);
  const how = idle
    ? "Keep your hands in view and move them as you would between signs, without making one. Record a few takes with your hands resting too."
    : sign && MOVING.has(sign.gloss)
      ? "Start from the first handshape and trace the whole movement during the take."
      : "Have the sign ready before the countdown ends and keep it up for the whole take.";
  const prompt = {
    idle: `Press Space or Record. ${how}`,
    countdown: `Get the sign ready… ${seconds}`,
    recording: "Recording: keep signing.",
    saving: "Saving…",
  }[phase];

  return (
    <Shell>
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm opacity-60">Sign to record</p>
          <p className="text-6xl font-semibold leading-none">{sign ? name(sign) : "–"}</p>
        </div>
        <p className="text-sm opacity-70">
          {done} of {TARGET_PER_SIGN} for this sign · {total} clips in total
        </p>
      </div>

      <div className={phase === "recording" ? "rounded-xl ring-4 ring-red-500" : ""}>
        <CameraTracker onFrame={onFrame} />
      </div>

      <p role="status" className="text-lg font-medium">
        {prompt}
      </p>
      {message && (
        <p role="alert" className="text-sm text-red-600">
          {message}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        <button onClick={() => step(-1)} className="rounded-md border px-4 py-2 text-sm">
          Previous
        </button>
        <button
          onClick={record}
          disabled={phase !== "idle" || !sign}
          className="rounded-md bg-foreground px-6 py-2 text-sm font-medium text-background disabled:opacity-40"
        >
          Record
        </button>
        <button onClick={() => step(1)} className="rounded-md border px-4 py-2 text-sm">
          Next
        </button>
        <button
          onClick={undo}
          disabled={!last || phase !== "idle"}
          className="rounded-md border px-4 py-2 text-sm disabled:opacity-40"
        >
          Delete last take
        </button>
      </div>

      <ul className="flex flex-wrap gap-1.5 text-xs">
        {signs.map((s, i) => {
          const n = counts[s.gloss] ?? 0;
          return (
            <li key={s.id}>
              <button
                onClick={() => setIndex(i)}
                aria-current={i === index}
                className={`rounded border px-2 py-1 ${i === index ? "bg-foreground text-background" : ""} ${n >= TARGET_PER_SIGN ? "border-green-600" : ""}`}
              >
                {name(s)} {n}
              </button>
            </li>
          );
        })}
      </ul>

      <p className="text-sm opacity-70">
        Each take is {CLIP_SECONDS} seconds. Between takes, change your distance, position and hand
        angle a little. Only landmark coordinates are saved; no video leaves this tab.
      </p>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 px-4 py-10">
      <Link href="/" className="text-sm underline underline-offset-4">
        Back
      </Link>
      <h1 className="text-2xl font-semibold tracking-tight">Record training clips</h1>
      {children}
    </main>
  );
}
