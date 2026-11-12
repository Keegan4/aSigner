"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CameraTracker } from "@/components/camera/CameraTracker";
import { FRAME_SIZE, LEFT_HAND, POSE } from "@/lib/tracking/features";
import {
  addSign,
  deleteTake,
  listSigns,
  removeSign,
  saveAccessCode,
  savedAccessCode,
  saveTake,
  TrainingApiError,
  type SavedTake,
  type TrainingSign,
} from "@/lib/training";

type Phase = "idle" | "countdown" | "recording" | "saving";
type Access = "loading" | "open" | "needs-code" | "closed" | "outdated" | "offline";

const COUNTDOWN_SECONDS = 3;
const CLIP_SECONDS = 2;
const TARGET_PER_SIGN = 25;

const hasHand = (frame: Float32Array) => frame.subarray(LEFT_HAND, POSE).some((v) => v !== 0);
const name = (s: TrainingSign) => (s.source === "idle" ? "No sign" : s.label);
// Fingerspelled letters other than J and Z are held shapes; everything else moves.
const moves = (s: TrainingSign) => s.label.length > 1 || s.label === "J" || s.label === "Z";

/**
 * Collects labelled landmark takes into the training database. Anyone with the
 * access code can record, and add signs that are not in the lessons yet.
 * Only landmark coordinates and the label are uploaded; no video leaves this tab.
 */
export function Trainer() {
  const [access, setAccess] = useState<Access>("loading");
  const [code, setCode] = useState("");
  const [signs, setSigns] = useState<TrainingSign[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [last, setLast] = useState<SavedTake | null>(null);
  const [adding, setAdding] = useState(false);

  const phaseRef = useRef<Phase>("idle");
  const buffer = useRef<Float32Array[]>([]);
  const startedAt = useRef(0);

  const sign = signs.find((s) => s.label === selected) ?? signs[0];

  const load = useCallback(async (tryCode: string) => {
    try {
      const rows = await listSigns(tryCode);
      setSigns(rows);
      setCode(tryCode);
      saveAccessCode(tryCode);
      setAccess("open");
      return true;
    } catch (err) {
      if (err instanceof TrainingApiError && err.status === 403) setAccess("needs-code");
      else if (err instanceof TrainingApiError && err.status === 503) setAccess("closed");
      else if (err instanceof TrainingApiError && err.status === 404) setAccess("outdated");
      else setAccess("offline");
      return false;
    }
  }, []);

  useEffect(() => {
    // Deferred so the stored code is read in the browser, after the first render.
    const timer = setTimeout(() => void load(savedAccessCode()), 0);
    return () => clearTimeout(timer);
  }, [load]);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  const refresh = useCallback(async () => {
    try {
      setSigns(await listSigns(code));
    } catch {
      // Counts are only a convenience; a failed refresh keeps the old ones.
    }
  }, [code]);

  const onFrame = useCallback((frame: Float32Array) => {
    if (phaseRef.current === "recording") buffer.current.push(frame);
  }, []);

  const upload = useCallback(
    async (target: TrainingSign) => {
      const frames = buffer.current;
      const elapsed = (performance.now() - startedAt.current) / 1000;
      if (frames.length < 5) {
        setMessage("Too few frames were captured. Check the camera and try again.");
        return;
      }
      // A resting take may have no hands in view at all, so the check is for signs only.
      if (target.source !== "idle" && frames.filter(hasHand).length < frames.length / 2) {
        setMessage("No hand was visible for most of that take, so it was not saved.");
        return;
      }
      const body = new Float32Array(frames.length * FRAME_SIZE);
      frames.forEach((frame, i) => body.set(frame, i * FRAME_SIZE));
      try {
        setLast(await saveTake(code, target.label, frames.length / elapsed, body));
        setSigns((all) =>
          all.map((s) =>
            s.label === target.label ? { ...s, total: s.total + 1, mine: s.mine + 1 } : s,
          ),
        );
        setMessage(null);
      } catch (err) {
        setMessage(
          `The take was not saved: ${err instanceof Error ? err.message : "the server is not reachable."}`,
        );
      }
    },
    [code],
  );

  const record = useCallback(() => {
    if (!sign || phaseRef.current !== "idle") return;
    const target = sign;
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
        await upload(target);
        setPhase("idle");
      }, CLIP_SECONDS * 1000);
    }, 1000);
  }, [sign, upload]);

  const undo = useCallback(async () => {
    if (!last) return;
    try {
      await deleteTake(code, last.id);
      setSigns((all) =>
        all.map((s) =>
          s.label === last.label
            ? { ...s, total: Math.max(0, s.total - 1), mine: Math.max(0, s.mine - 1) }
            : s,
        ),
      );
      setLast(null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "The take was not deleted.");
    }
  }, [code, last]);

  const step = useCallback(
    (by: number) => {
      if (!signs.length) return;
      const at = Math.max(0, signs.findIndex((s) => s.label === sign?.label));
      setSelected(signs[(at + by + signs.length) % signs.length].label);
    },
    [signs, sign],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Typing in the add-sign form or the filter must not start a take.
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

  const shown = useMemo(() => {
    const q = filter.trim().toUpperCase();
    return q ? signs.filter((s) => name(s).toUpperCase().includes(q)) : signs;
  }, [signs, filter]);

  if (access === "loading") return <Frame>Loading…</Frame>;
  if (access === "offline") {
    return <Frame>The aSigner server is not reachable. Check that it is running and reload.</Frame>;
  }
  if (access === "outdated") {
    return (
      <Frame>
        The aSigner server does not have the training API. Update it, run the database migration
        and restart it.
      </Frame>
    );
  }
  if (access === "closed") {
    return (
      <Frame>
        Training collection is not set up on this server yet. The owner needs to set
        TRAINING_ACCESS_CODE.
      </Frame>
    );
  }
  if (access === "needs-code") return <CodeForm onSubmit={load} />;

  const mine = sign?.mine ?? 0;
  const yours = signs.reduce((a, s) => a + s.mine, 0);
  const how = !sign
    ? ""
    : sign.source === "idle"
      ? "Keep your hands in view and move them as you would between signs, without making one. Record a few takes with your hands resting too."
      : moves(sign)
        ? "Start from the first handshape and make the whole movement during the take."
        : "Have the sign ready before the countdown ends and keep it up for the whole take.";
  const prompt = {
    idle: `Press Space or Record. ${how}`,
    countdown: `Get the sign ready… ${seconds}`,
    recording: "Recording: keep signing.",
    saving: "Saving…",
  }[phase];

  return (
    <Frame>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm opacity-60">Sign to record</p>
          <p className="text-6xl font-semibold leading-none break-all">{sign ? name(sign) : "–"}</p>
        </div>
        <p className="text-sm opacity-70">
          {mine} of {TARGET_PER_SIGN} from you · {sign?.total ?? 0} from everyone · {yours}{" "}
          {yours === 1 ? "take" : "takes"} by you in total
        </p>
      </div>
      {sign?.description && (
        <p className="rounded-md border px-3 py-2 text-sm">
          <span className="opacity-60">How it is made: </span>
          {sign.description}
        </p>
      )}

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

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="mr-auto text-lg font-semibold">Signs</h2>
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Find a sign"
            aria-label="Find a sign"
            className="rounded-md border bg-transparent px-3 py-1.5 text-sm"
          />
          <button
            onClick={() => setAdding((a) => !a)}
            aria-expanded={adding}
            className="rounded-md border px-3 py-1.5 text-sm"
          >
            {adding ? "Close" : "Add a sign"}
          </button>
        </div>

        {adding && (
          <AddSignForm
            onAdd={async (label, description) => {
              const made = await addSign(code, label, description);
              await refresh();
              setSelected(made.label);
              setFilter("");
              setAdding(false);
            }}
          />
        )}

        <ul className="flex flex-wrap gap-1.5 text-xs">
          {shown.map((s) => (
            <li key={s.label}>
              <button
                onClick={() => setSelected(s.label)}
                aria-current={s.label === sign?.label}
                title={s.description || undefined}
                className={`rounded border px-2 py-1 ${s.label === sign?.label ? "bg-foreground text-background" : ""} ${s.mine >= TARGET_PER_SIGN ? "border-green-600" : ""} ${s.source === "custom" ? "border-dashed" : ""}`}
              >
                {name(s)} {s.mine}
                {s.total > s.mine && <span className="opacity-60"> / {s.total}</span>}
              </button>
            </li>
          ))}
          {!shown.length && <li className="opacity-60">No sign matches “{filter}”.</li>}
        </ul>
        <p className="text-xs opacity-60">
          Each button shows your takes, then everyone&apos;s. Dashed signs were added on this page.
        </p>

        {sign?.source === "custom" && sign.total === 0 && (
          <button
            onClick={async () => {
              try {
                await removeSign(code, sign.label);
                setSelected(null);
                await refresh();
              } catch (err) {
                setMessage(err instanceof Error ? err.message : "The sign was not removed.");
              }
            }}
            className="self-start text-sm underline underline-offset-4"
          >
            Remove {sign.label} (no takes yet)
          </button>
        )}
      </section>

      <p className="text-sm opacity-70">
        Each take is {CLIP_SECONDS} seconds. Between takes, change your distance, position and hand
        angle a little. Only landmark coordinates and the sign&apos;s name are saved, with a random
        id for this browser so your own takes can be counted and undone. No video leaves this tab,
        and no name or account is attached.
      </p>
    </Frame>
  );
}

function AddSignForm({ onAdd }: { onAdd: (label: string, description: string) => Promise<void> }) {
  const [label, setLabel] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  return (
    <form
      className="flex flex-col gap-2 rounded-lg border p-3"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setError(null);
        try {
          await onAdd(label, description);
        } catch (err) {
          setError(err instanceof Error ? err.message : "The sign was not added.");
        } finally {
          setBusy(false);
        }
      }}
    >
      <label className="flex flex-col gap-1 text-sm">
        Name
        <input
          required
          maxLength={40}
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="For example: thank you"
          className="rounded-md border bg-transparent px-3 py-1.5"
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        How it is made (optional, helps others record it the same way)
        <textarea
          maxLength={300}
          rows={2}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="rounded-md border bg-transparent px-3 py-1.5"
        />
      </label>
      <p className="text-xs opacity-60">
        Names are saved in capitals with hyphens for spaces, so “thank you” becomes THANK-YOU.
      </p>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
      <button
        disabled={busy || !label.trim()}
        className="self-start rounded-md bg-foreground px-4 py-1.5 text-sm font-medium text-background disabled:opacity-40"
      >
        Add sign
      </button>
    </form>
  );
}

function CodeForm({ onSubmit }: { onSubmit: (code: string) => Promise<boolean> }) {
  const [value, setValue] = useState("");
  const [wrong, setWrong] = useState(false);

  return (
    <Frame>
      <form
        className="flex flex-col gap-3"
        onSubmit={async (e) => {
          e.preventDefault();
          setWrong(!(await onSubmit(value.trim())));
        }}
      >
        <label className="flex flex-col gap-1 text-sm">
          Access code
          <input
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            className="rounded-md border bg-transparent px-3 py-2"
          />
        </label>
        {wrong && (
          <p role="alert" className="text-sm text-red-600">
            That code was not accepted.
          </p>
        )}
        <button className="self-start rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background">
          Continue
        </button>
        <p className="text-sm opacity-70">Ask the person who sent you this page for the code.</p>
      </form>
    </Frame>
  );
}

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 px-4 py-10">
      <Link href="/" className="text-sm underline underline-offset-4">
        Back
      </Link>
      <h1 className="text-2xl font-semibold tracking-tight">Help train the sign model</h1>
      {children}
    </main>
  );
}
