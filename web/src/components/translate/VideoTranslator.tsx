"use client";

import { DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

import { Button, TextLink } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { EmptyState } from "@/components/ui/EmptyState";
import { Field, FileInput, Select } from "@/components/ui/Field";
import { GlossTokens } from "@/components/ui/Gloss";
import { IconClose, IconDownload, IconFilm } from "@/components/ui/Icons";
import { Meter } from "@/components/ui/Meter";
import { ModelNote } from "@/components/ui/ModelNote";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Shell } from "@/components/ui/Shell";
import { SignInPrompt } from "@/components/ui/States";
import { Switch } from "@/components/ui/Switch";
import { type ModelInfo, WS_URL } from "@/lib/api";
import { FRAME_SIZE, packFrame } from "@/lib/tracking/features";
import { closeTrackers, detect, loadTrackers, type Trackers } from "@/lib/tracking/landmarkers";

/** Messages of the server's /ws/translate-video endpoint (server/app/ws/translate.py). */
type Message =
  | { type: "ready"; model: ModelInfo | null }
  | { type: "window"; t: number; label: string | null; top: [string, number][] }
  | { type: "gloss"; t: number; gloss: string; glosses: string[] }
  | { type: "sentence"; t: number; glosses: string[]; sentence: string; from_model: boolean; id?: number }
  | { type: "done" }
  | { type: "error"; code: string; message: string };

type Window = Extract<Message, { type: "window" }>;
type Gloss = Extract<Message, { type: "gloss" }>;
type Sentence = Extract<Message, { type: "sentence" }>;

type Status =
  | { kind: "empty" }
  | { kind: "loaded" }
  | { kind: "running"; progress: number }
  | { kind: "done" }
  | { kind: "stopped" }
  | { kind: "error"; message: string }
  | { kind: "signed-out" };

// The video is drawn at most this wide for tracking; larger frames only slow MediaPipe down.
const MAX_WIDTH = 640;
const RATES = [10, 15, 30];

const time = (t: number) => `${t.toFixed(2)}s`;
const percent = (p: number) => `${Math.round(p * 100)}%`;

/** Waits for the video to show the frame at `t` seconds. */
function seek(video: HTMLVideoElement, t: number) {
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", fail);
      reject(new Error("The video could not be read."));
    };
    video.addEventListener("seeked", done);
    video.addEventListener("error", fail);
    video.currentTime = t;
  });
}

/**
 * Translates a recorded video instead of a live camera: a debugging aid for
 * the sign model. The video is stepped through frame by frame in the browser
 * and only landmark coordinates, each with its time in the video, are sent.
 */
export function VideoTranslator() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackers = useRef<Trackers | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const cancel = useRef(false);
  // MediaPipe's video mode needs timestamps that always increase, across runs too.
  const lastMs = useRef(0);

  const [url, setUrl] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "empty" });
  const [rate, setRate] = useState(15);
  const [mirror, setMirror] = useState(false);
  const [save, setSave] = useState(false);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [windows, setWindows] = useState<Window[]>([]);
  const [glosses, setGlosses] = useState<Gloss[]>([]);
  const [sentences, setSentences] = useState<Sentence[]>([]);

  useEffect(() => {
    return () => {
      cancel.current = true;
      socket.current?.close();
      if (trackers.current) closeTrackers(trackers.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  function choose(file: File | undefined) {
    if (!file) return;
    stop();
    setUrl(URL.createObjectURL(file));
    setName(file.name);
    setStatus({ kind: "loaded" });
    setWindows([]);
    setGlosses([]);
    setSentences([]);
  }

  /** Draws the current video frame (mirrored if asked) onto the canvas and returns its context. */
  function paint() {
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d")!;
    ctx.save();
    if (mirror) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    return ctx;
  }

  async function show(t: number) {
    const video = videoRef.current;
    if (!video || status.kind === "running") return;
    await seek(video, t);
    paint();
  }

  function connect() {
    return new Promise<WebSocket>((resolve, reject) => {
      const ws = new WebSocket(`${WS_URL}/ws/translate-video${save ? "?save=true" : ""}`);
      ws.binaryType = "arraybuffer";
      let ready = false;
      ws.onmessage = (e) => {
        const message: Message = JSON.parse(e.data);
        switch (message.type) {
          case "ready":
            ready = true;
            setModel(message.model);
            resolve(ws);
            break;
          case "window":
            setWindows((list) => [...list, message]);
            break;
          case "gloss":
            setGlosses((list) => [...list, message]);
            break;
          case "sentence":
            setSentences((list) => [...list, message]);
            break;
          case "done":
            setStatus({ kind: "done" });
            ws.close();
            break;
          case "error":
            reject(new Error(message.message));
            break;
        }
      };
      ws.onclose = (e) => {
        if (e.code === 4401) reject(new Error("signed-out"));
        else if (!ready) reject(new Error("The aSigner server is not reachable. Start it and try again."));
      };
    });
  }

  async function run() {
    const video = videoRef.current;
    if (!video || !url) return;
    cancel.current = false;
    setWindows([]);
    setGlosses([]);
    setSentences([]);
    setStatus({ kind: "running", progress: 0 });
    try {
      const [ws, loaded] = await Promise.all([
        connect(),
        trackers.current ? Promise.resolve(trackers.current) : loadTrackers(),
      ]);
      socket.current = ws;
      trackers.current = loaded;
      const draw = new DrawingUtils(canvasRef.current!.getContext("2d")!);
      const duration = video.duration;
      const base = lastMs.current + 1000;

      for (let t = 0; t <= duration; t += 1 / rate) {
        if (cancel.current) return;
        await seek(video, t);
        const ctx = paint();
        const timeMs = base + t * 1000;
        lastMs.current = timeMs;
        const result = detect(loaded, ctx.canvas, timeMs);
        for (const hand of result.hands) {
          draw.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#f59e0b", lineWidth: 2 });
          draw.drawLandmarks(hand, { color: "#fef3c7", radius: 2 });
        }
        const message = new ArrayBuffer(FRAME_SIZE * 4 + 8);
        new Float32Array(message, 0, FRAME_SIZE).set(packFrame(result));
        new DataView(message).setFloat64(FRAME_SIZE * 4, t, true);
        if (ws.readyState !== WebSocket.OPEN) throw new Error("The connection to the aSigner server was lost.");
        ws.send(message);
        setStatus({ kind: "running", progress: duration ? t / duration : 1 });
      }
      ws.send(JSON.stringify({ type: "end" }));
    } catch (err) {
      socket.current?.close();
      const message = err instanceof Error ? err.message : String(err);
      setStatus(message === "signed-out" ? { kind: "signed-out" } : { kind: "error", message });
    }
  }

  function stop() {
    cancel.current = true;
    socket.current?.close();
    socket.current = null;
    setStatus((s) => (s.kind === "running" ? { kind: "stopped" } : s));
  }

  function download() {
    const log = { video: name, rate, mirror, model, windows, glosses, sentences };
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(log, null, 2)], { type: "application/json" }));
    link.download = `${name.replace(/\.[^.]+$/, "") || "video"}-translation.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  const running = status.kind === "running";
  const finished = status.kind === "done" || status.kind === "stopped";

  return (
    <Shell
      back="/translate"
      backLabel="Live translation"
      title="Translate a video"
      width="wide"
      description={
        <>
          Choose a recording of someone signing to see what the sign model reads, window by window. The video
          stays in your browser: it is tracked here and only landmark coordinates are sent.
        </>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[22rem_minmax(0,1fr)] lg:items-start">
        <Card as="section" aria-label="Video and options" className="flex flex-col gap-5 text-sm">
          <Step number={1} title="Choose a video">
            <Field label="Video file" hint={name ? `Loaded: ${name}` : "Any video your browser can play."}>
              <FileInput accept="video/*" disabled={running} onChange={(e) => choose(e.target.files?.[0])} />
            </Field>
          </Step>
          <Step number={2} title="Options">
            <Field label="Frames per second">
              <Select value={rate} disabled={running} onChange={(e) => setRate(Number(e.target.value))}>
                {RATES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </Select>
            </Field>
            <div className="flex flex-col">
              <Switch
                label="Mirror the video"
                checked={mirror}
                disabled={running}
                onChange={(e) => setMirror(e.target.checked)}
              />
              <Switch
                label="Save sentences to my history"
                checked={save}
                disabled={running}
                onChange={(e) => setSave(e.target.checked)}
              />
            </div>
            <p className="text-xs text-muted">
              Mirror the video if it was recorded with a selfie camera that saves a mirrored picture; the live
              translator tracks the camera unmirrored. Sentences are only kept when you ask.
            </p>
          </Step>
          <Step number={3} title="Translate">
            <div className="flex flex-wrap items-center gap-2">
              {running ? (
                <Button onClick={stop} icon={<IconClose size={18} />}>
                  Stop
                </Button>
              ) : (
                <Button variant="primary" onClick={run} disabled={!url}>
                  {finished ? "Translate again" : "Translate"}
                </Button>
              )}
              {(finished || running) && windows.length > 0 && (
                <Button onClick={download} icon={<IconDownload size={18} />}>
                  Download log (JSON)
                </Button>
              )}
            </div>
            {running && (
              <div className="flex items-center gap-3">
                <Meter
                  kind="progress"
                  value={status.progress}
                  max={1}
                  label="Translation progress"
                  valueText={percent(status.progress)}
                />
                <span className="w-10 text-right text-xs text-muted tabular-nums">{percent(status.progress)}</span>
              </div>
            )}
            {!url && <p className="text-xs text-muted">Choose a video first.</p>}
          </Step>
        </Card>

        <div className="flex min-w-0 flex-col gap-6">
          {status.kind === "error" && (
            <Callout tone="danger" role="alert">
              {status.message}
            </Callout>
          )}
          {status.kind === "signed-out" && <SignInPrompt role="alert" to="translate" />}

          {!url && (
            <EmptyState icon={<IconFilm size={24} />} title="No video chosen">
              The video appears here with the tracking drawn on it.
            </EmptyState>
          )}
          <div className={`relative w-full overflow-hidden rounded-card bg-stage ${url ? "" : "hidden"}`}>
            {/* The video is only a frame source; frames are shown on the canvas with the tracking on top. */}
            <video
              ref={videoRef}
              src={url ?? undefined}
              muted
              playsInline
              preload="auto"
              onLoadedData={() => show(0)}
              className="hidden"
            />
            <canvas ref={canvasRef} className="h-auto w-full" />
          </div>

          {model && <ModelNote model={model} />}

          {(glosses.length > 0 || sentences.length > 0 || finished) && (
            <section aria-labelledby="video-translation" className="flex flex-col gap-3">
              <SectionHeading id="video-translation" variant="eyebrow">
                Translation
              </SectionHeading>
              {sentences.length === 0 && glosses.length === 0 ? (
                <p className="text-sm text-muted">No signs were recognised.</p>
              ) : (
                <>
                  <ul aria-live="polite" className="flex flex-col gap-2">
                    {sentences.map((s, i) => (
                      <Card as="li" key={i} padding="sm" className="flex flex-col gap-2 sm:px-4">
                        <p className="text-lg font-medium">{s.sentence}</p>
                        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted">
                          <GlossTokens glosses={s.glosses} size="sm" />
                          <span className="tabular-nums">
                            · ended at {time(s.t)}
                            {s.from_model ? "" : " · plain glosses"}
                          </span>
                        </p>
                      </Card>
                    ))}
                  </ul>
                  <div className="flex flex-col gap-2">
                    <p className="text-xs text-muted">Each sign as it was read. Choose one to show that frame.</p>
                    <ol className="flex flex-wrap gap-2">
                      {glosses.map((g, i) => (
                        <li key={i}>
                          <Chip
                            onClick={() => show(g.t)}
                            disabled={running}
                            title="Show this frame"
                            className="font-mono"
                          >
                            {g.gloss} <span className="font-sans text-xs font-normal text-muted">{time(g.t)}</span>
                          </Chip>
                        </li>
                      ))}
                    </ol>
                  </div>
                </>
              )}
            </section>
          )}

          {windows.length > 0 && (
            <section aria-labelledby="video-windows" className="flex flex-col gap-3">
              <SectionHeading
                id="video-windows"
                variant="eyebrow"
                description="Each row is one look at the last second. A sign is counted once it is read the same way several times running. A dash means too little seen yet, no hands, or not confident enough."
              >
                What the model saw ({windows.length} windows)
              </SectionHeading>
              <div className="max-h-96 overflow-auto rounded-card border border-line bg-surface">
                <table className="w-full text-left font-mono text-sm">
                  <thead className="sticky top-0 bg-sunken text-xs text-muted">
                    <tr className="border-b border-line">
                      <th className="px-3 py-2 font-medium">Time</th>
                      <th className="px-3 py-2 font-medium">Read as</th>
                      <th className="px-3 py-2 font-medium">Top guesses</th>
                    </tr>
                  </thead>
                  <tbody>
                    {windows.map((w, i) => (
                      <tr
                        key={i}
                        onClick={() => show(w.t)}
                        className={`cursor-pointer border-b border-line last:border-0 hover:bg-sunken ${w.label ? "text-foreground" : "text-muted"}`}
                      >
                        <td className="px-3 py-1.5 tabular-nums">{time(w.t)}</td>
                        <td className="px-3 py-1.5 font-semibold">{w.label ?? "–"}</td>
                        <td className="px-3 py-1.5">
                          {w.top.length ? w.top.map(([g, p]) => `${g} ${percent(p)}`).join(", ") : "–"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <p className="text-sm">
            <TextLink href="/translate">Translate live from the camera</TextLink>
          </p>
        </div>
      </div>
    </Shell>
  );
}

function Step({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 border-b border-line pb-5 last:border-0 last:pb-0">
      <h2 className="flex items-center gap-2.5 text-base font-semibold">
        <span
          aria-hidden
          className="flex size-6 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent-fg"
        >
          {number}
        </span>
        <span>
          <span className="sr-only">Step {number}: </span>
          {title}
        </span>
      </h2>
      {children}
    </div>
  );
}
