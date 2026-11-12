"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CameraTracker } from "@/components/camera/CameraTracker";
import {
  emptyFeed,
  type FrameFeed,
  landmarkCount,
  LandmarksPanel,
  type Prediction,
  PredictionsPanel,
} from "@/components/translate/DebugPanels";
import { Shell } from "@/components/ui/Shell";
import { ModelNote } from "@/components/ui/ModelNote";
import { type ModelInfo, type Translation, WS_URL } from "@/lib/api";

type Link_ = "connecting" | "open" | "signed-out" | "closed";

/** Messages of the server's /ws/translate endpoint (server/app/ws/translate.py). */
type Message =
  | { type: "ready"; model: ModelInfo | null }
  | ({ type: "window" } & Prediction)
  | { type: "gloss"; gloss: string; glosses: string[] }
  | ({ type: "sentence" } & Translation)
  | { type: "cleared" }
  | { type: "error"; code: string; message: string };

export function Translator() {
  const socket = useRef<WebSocket | null>(null);
  const speakRef = useRef(false);
  const [link, setLink] = useState<Link_>("connecting");
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [glosses, setGlosses] = useState<string[]>([]);
  const [sentences, setSentences] = useState<Translation[]>([]);
  const [speak, setSpeak] = useState(false);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [debug, setDebug] = useState(true);
  const feed = useRef<FrameFeed>(emptyFeed());

  useEffect(() => {
    speakRef.current = speak;
    if (!speak) window.speechSynthesis?.cancel();
  }, [speak]);

  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/translate`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      const message: Message = JSON.parse(e.data);
      switch (message.type) {
        case "ready":
          setModel(message.model);
          setLink("open");
          break;
        case "window":
          setPrediction({ label: message.label, top: message.top });
          break;
        case "gloss":
          setGlosses(message.glosses);
          break;
        case "sentence":
          setGlosses([]);
          setSentences((list) => [message, ...list]);
          if (speakRef.current && "speechSynthesis" in window) {
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(message.sentence));
          }
          break;
        case "cleared":
          setGlosses([]);
          break;
        case "error":
          setUnavailable(message.message);
          break;
      }
    };
    ws.onclose = (e) => setLink(e.code === 4401 ? "signed-out" : "closed");
    socket.current = ws;
    return () => {
      ws.onclose = null;
      ws.close();
      window.speechSynthesis?.cancel();
    };
  }, []);

  const onFrame = useCallback((frame: Float32Array) => {
    feed.current.frame = frame;
    feed.current.sent++;
    feed.current.points += landmarkCount(frame);
    const ws = socket.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(frame);
  }, []);

  const onVideoFrame = useCallback((presentedFrames: number) => {
    feed.current.videoFrames = presentedFrames;
  }, []);
  const onCamera = useCallback((settings: MediaTrackSettings) => {
    feed.current.cameraFps = settings.frameRate;
  }, []);

  const send = (type: "flush" | "clear") => {
    const ws = socket.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type }));
  };

  return (
    <Shell back="/" title="Translate" wide={debug}>
      <label className="flex items-center gap-2 self-start text-sm">
        <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
        Show debug panels (model predictions and landmarks)
      </label>
      <div
        className={
          debug
            ? "grid gap-6 lg:grid-cols-[16rem_minmax(0,1fr)_18rem] lg:items-start"
            : "flex flex-col gap-6"
        }
      >
        {debug && (
          <aside className="order-2 lg:order-1 lg:sticky lg:top-4">
            <PredictionsPanel prediction={prediction} />
          </aside>
        )}
        <div className="order-1 flex min-w-0 flex-col gap-6 lg:order-2">
          <CameraTracker onFrame={onFrame} onVideoFrame={onVideoFrame} onCamera={onCamera} />

          {unavailable ? (
            <p role="alert" className="rounded-md border p-3 text-sm">
              {unavailable}
            </p>
          ) : link === "signed-out" ? (
            <p role="alert" className="text-sm">
              <a href="/api/auth/login" className="underline underline-offset-4">
                Sign in with Google
              </a>{" "}
              to translate.
            </p>
          ) : link === "closed" ? (
            <p role="alert" className="text-sm text-red-600">
              The connection to the aSigner server was lost. Reload to try again.
            </p>
          ) : (
            <>
              <section className="flex flex-col gap-3 rounded-lg border p-4">
                <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">Signs so far</h2>
                <p aria-live="polite" className="min-h-8 font-mono text-xl">
                  {glosses.length ? glosses.join(" ") : <span className="text-sm opacity-60">
                    {link === "open" ? "Sign to the camera. Pause to end a sentence." : "Connecting…"}
                  </span>}
                </p>
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <button
                    onClick={() => send("flush")}
                    disabled={glosses.length === 0}
                    className="rounded-md bg-foreground px-3 py-1.5 font-medium text-background disabled:opacity-40"
                  >
                    End sentence
                  </button>
                  <button
                    onClick={() => send("clear")}
                    disabled={glosses.length === 0}
                    className="rounded-md border px-3 py-1.5 disabled:opacity-40"
                  >
                    Clear
                  </button>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
                    Speak sentences aloud
                  </label>
                </div>
              </section>

              <ModelNote model={model} />

              <section className="flex flex-col gap-3">
                <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">Sentences</h2>
                {sentences.length === 0 ? (
                  <p className="text-sm opacity-70">Finished sentences appear here and are saved.</p>
                ) : (
                  <ul aria-live="polite" className="flex flex-col gap-2">
                    {sentences.map((s) => (
                      <li key={s.id} className="rounded-lg border p-3">
                        <p className="text-lg">{s.sentence}</p>
                        <p className="font-mono text-sm opacity-70">{s.glosses.join(" ")}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          )}

          <p className="text-sm opacity-70">
            Only landmark coordinates are sent, and only the text of each sentence is kept.{" "}
            <Link href="/translations" className="underline underline-offset-4">
              Past translations
            </Link>
            {" · "}
            <Link href="/translate/video" className="underline underline-offset-4">
              Translate a video instead
            </Link>
          </p>
        </div>
        {debug && (
          <aside className="order-3 lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto">
            <LandmarksPanel feed={feed} />
          </aside>
        )}
      </div>
    </Shell>
  );
}
