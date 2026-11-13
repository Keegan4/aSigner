"use client";

import { DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ModelNote } from "@/components/ui/ModelNote";
import { Shell } from "@/components/ui/Shell";
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
        new Float32Array(message, 0, FRAME_SIZE).set(packFrame(result, ctx.canvas.width, ctx.canvas.height));
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
    <Shell back="/translate" backLabel="Live translation" title="Translate a video">
      <p className="text-sm opacity-70">
        Choose a recording of someone signing to see what the sign model reads, window by window. The video
        stays in your browser: it is tracked here and only landmark coordinates are sent.
      </p>

      <section className="flex flex-col gap-3 rounded-lg border p-4 text-sm">
        <input
          type="file"
          accept="video/*"
          disabled={running}
          onChange={(e) => choose(e.target.files?.[0])}
          className="file:mr-3 file:rounded-md file:border file:bg-transparent file:px-3 file:py-1.5"
        />
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2">
            Frames per second
            <select
              value={rate}
              disabled={running}
              onChange={(e) => setRate(Number(e.target.value))}
              className="rounded-md border bg-transparent px-2 py-1"
            >
              {RATES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={mirror} disabled={running} onChange={(e) => setMirror(e.target.checked)} />
            Mirror the video
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={save} disabled={running} onChange={(e) => setSave(e.target.checked)} />
            Save sentences to my history
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          {running ? (
            <button onClick={stop} className="rounded-md border px-3 py-1.5">
              Stop
            </button>
          ) : (
            <button
              onClick={run}
              disabled={!url}
              className="rounded-md bg-foreground px-3 py-1.5 font-medium text-background disabled:opacity-40"
            >
              {finished ? "Translate again" : "Translate"}
            </button>
          )}
          {(finished || running) && windows.length > 0 && (
            <button onClick={download} className="rounded-md border px-3 py-1.5">
              Download log (JSON)
            </button>
          )}
          {running && (
            <progress value={status.progress} max={1} className="flex-1" aria-label="Translation progress" />
          )}
        </div>
        <p className="text-xs opacity-60">
          Mirror the video if it was recorded with a selfie camera that saves a mirrored picture; the live
          translator tracks the camera unmirrored. Sentences are only kept when you ask.
        </p>
      </section>

      {status.kind === "error" && (
        <p role="alert" className="rounded-md border p-3 text-sm">
          {status.message}
        </p>
      )}
      {status.kind === "signed-out" && (
        <p role="alert" className="text-sm">
          <a href="/api/auth/login" className="underline underline-offset-4">
            Sign in with Google
          </a>{" "}
          to translate.
        </p>
      )}

      <div className={`relative w-full overflow-hidden rounded-xl bg-neutral-900 ${url ? "" : "hidden"}`}>
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
        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">Translation</h2>
          {sentences.length === 0 && glosses.length === 0 ? (
            <p className="text-sm opacity-70">No signs were recognised.</p>
          ) : (
            <>
              <ul aria-live="polite" className="flex flex-col gap-2">
                {sentences.map((s, i) => (
                  <li key={i} className="rounded-lg border p-3">
                    <p className="text-lg">{s.sentence}</p>
                    <p className="font-mono text-sm opacity-70">
                      {s.glosses.join(" ")} · ended at {time(s.t)}
                      {s.from_model ? "" : " · plain glosses"}
                    </p>
                  </li>
                ))}
              </ul>
              <ol className="flex flex-wrap gap-2 font-mono text-sm">
                {glosses.map((g, i) => (
                  <li key={i}>
                    <button
                      onClick={() => show(g.t)}
                      disabled={running}
                      className="rounded-md border px-2 py-1 hover:bg-foreground/5"
                      title="Show this frame"
                    >
                      {g.gloss} <span className="opacity-60">{time(g.t)}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>
      )}

      {windows.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">
            What the model saw ({windows.length} windows)
          </h2>
          <p className="text-xs opacity-60">
            Each row is one look at the last two seconds. A sign is counted once it is read the same way several
            times running. A dash means too little seen yet, no hands, or not confident enough.
          </p>
          <div className="max-h-96 overflow-auto rounded-lg border">
            <table className="w-full text-left font-mono text-sm">
              <thead className="sticky top-0 bg-background">
                <tr className="border-b">
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
                    className={`cursor-pointer border-b last:border-0 hover:bg-foreground/5 ${w.label ? "" : "opacity-60"}`}
                  >
                    <td className="px-3 py-1.5">{time(w.t)}</td>
                    <td className="px-3 py-1.5">{w.label ?? "–"}</td>
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

      <p className="text-sm opacity-70">
        <Link href="/translate" className="underline underline-offset-4">
          Translate live from the camera
        </Link>
      </p>
    </Shell>
  );
}
