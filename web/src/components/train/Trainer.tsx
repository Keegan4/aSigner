"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CameraStage, type StageOverlay } from "@/components/camera/CameraStage";
import { CameraTracker } from "@/components/camera/CameraTracker";
import { CaptureControls, CaptureSign, SignChips } from "@/components/camera/CapturePanel";
import { Button } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { Field, Input, SearchInput, Textarea } from "@/components/ui/Field";
import { IconClose, IconPlus, IconTrash } from "@/components/ui/Icons";
import { Shell } from "@/components/ui/Shell";
import { LoadingBlock, ServerUnreachable } from "@/components/ui/States";
import { CLIP_SECONDS, FRAME_SIZE, LEFT_HAND, POSE } from "@/lib/tracking/features";
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
// A sign's take is kept when a hand was tracked in at least this share of its
// frames. Low, because tracking drops out while a hand moves fast.
const MIN_HAND_SHARE = 0.25;
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
      if (
        target.source !== "idle" &&
        frames.filter(hasHand).length < frames.length * MIN_HAND_SHARE
      ) {
        setMessage("Hardly any of that take showed a hand, so it was not saved.");
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

  if (access === "loading") {
    return (
      <Frame>
        <LoadingBlock lines={4} />
      </Frame>
    );
  }
  if (access === "offline") {
    return (
      <Frame>
        <ServerUnreachable message="The aSigner server is not reachable. Check that it is running and reload." />
      </Frame>
    );
  }
  if (access === "outdated") {
    return (
      <Frame>
        <Callout tone="warning" title="The server needs updating">
          The aSigner server does not have the training API. Update it, run the database migration
          and restart it.
        </Callout>
      </Frame>
    );
  }
  if (access === "closed") {
    return (
      <Frame>
        <Callout tone="info" title="Not open yet">
          Training collection is not set up on this server yet. The owner needs to set
          TRAINING_ACCESS_CODE.
        </Callout>
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
            count={mine}
            target={TARGET_PER_SIGN}
            countText={`${mine} of ${TARGET_PER_SIGN} from you`}
            details={
              <>
                {sign?.total ?? 0} from everyone · {yours} {yours === 1 ? "take" : "takes"} by you in total
              </>
            }
          >
            {sign?.description && (
              <Callout tone="neutral" icon={false}>
                <span className="text-muted">How it is made: </span>
                {sign.description}
              </Callout>
            )}
          </CaptureSign>
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
            angle a little. Only landmark coordinates and the sign&apos;s name are saved, with a random
            id for this browser so your own takes can be counted and undone. No video leaves this tab,
            and no name or account is attached.
          </p>
        </div>

        <Card
          as="section"
          aria-labelledby="train-signs"
          className="flex flex-col gap-4 lg:col-start-2 lg:row-start-2 lg:self-start"
        >
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="train-signs" className="mr-auto text-lg font-semibold tracking-tight">
              Signs
            </h2>
            <Button
              size="sm"
              onClick={() => setAdding((a) => !a)}
              aria-expanded={adding}
              icon={adding ? <IconClose size={16} /> : <IconPlus size={16} />}
            >
              {adding ? "Close" : "Add a sign"}
            </Button>
          </div>
          <SearchInput
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Find a sign"
            aria-label="Find a sign"
          />

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

          <SignChips
            items={shown.map((s) => ({
              key: s.label,
              label: name(s),
              count: s.mine,
              extra: s.total > s.mine ? ` / ${s.total}` : undefined,
              current: s.label === sign?.label,
              done: s.mine >= TARGET_PER_SIGN,
              dashed: s.source === "custom",
              title: s.description || undefined,
            }))}
            onSelect={setSelected}
            empty={<>No sign matches “{filter}”.</>}
          />
          <p className="text-xs text-muted">
            Each button shows your takes, then everyone&apos;s. Dashed signs were added on this page.
          </p>

          {sign?.source === "custom" && sign.total === 0 && (
            <Button
              size="sm"
              variant="danger-ghost"
              icon={<IconTrash size={16} />}
              className="self-start"
              onClick={async () => {
                try {
                  await removeSign(code, sign.label);
                  setSelected(null);
                  await refresh();
                } catch (err) {
                  setMessage(err instanceof Error ? err.message : "The sign was not removed.");
                }
              }}
            >
              Remove {sign.label} (no takes yet)
            </Button>
          )}
        </Card>
      </div>
    </Frame>
  );
}

function AddSignForm({ onAdd }: { onAdd: (label: string, description: string) => Promise<void> }) {
  const [label, setLabel] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  return (
    <Card
      as="form"
      tone="sunken"
      padding="sm"
      className="flex flex-col gap-3 sm:p-4"
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
      <Field label="Name" hint="Names are saved in capitals with hyphens for spaces, so “thank you” becomes THANK-YOU.">
        <Input
          required
          maxLength={40}
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="For example: thank you"
        />
      </Field>
      <Field label="How it is made" optional hint="Helps others record it the same way.">
        <Textarea maxLength={300} rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />
      </Field>
      {error && (
        <Callout tone="danger" role="alert">
          {error}
        </Callout>
      )}
      <Button type="submit" variant="primary" loading={busy} disabled={!label.trim()} className="self-start">
        Add sign
      </Button>
    </Card>
  );
}

function CodeForm({ onSubmit }: { onSubmit: (code: string) => Promise<boolean> }) {
  const [value, setValue] = useState("");
  const [wrong, setWrong] = useState(false);

  return (
    <Frame>
      <Card
        as="form"
        padding="lg"
        className="flex max-w-md flex-col gap-4"
        onSubmit={async (e) => {
          e.preventDefault();
          setWrong(!(await onSubmit(value.trim())));
        }}
      >
        <Field label="Access code" error={wrong ? "That code was not accepted." : undefined}>
          <Input autoFocus value={value} onChange={(e) => setValue(e.target.value)} autoComplete="off" />
        </Field>
        <Button type="submit" variant="primary" className="self-start">
          Continue
        </Button>
        <p className="text-sm text-muted">Ask the person who sent you this page for the code.</p>
      </Card>
    </Frame>
  );
}

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <Shell back="/" backLabel="Lessons" title="Help train the sign model" width="default">
      {children}
    </Shell>
  );
}
