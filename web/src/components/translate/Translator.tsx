"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraStage, StagePill } from "@/components/camera/CameraStage";
import { CameraTracker } from "@/components/camera/CameraTracker";
import {
  emptyFeed,
  type FrameFeed,
  landmarkCount,
  LandmarksPanel,
  type Prediction,
  PredictionsPanel,
} from "@/components/translate/DebugPanels";
import { type LiveSeen, TrainedSignsPanel } from "@/components/translate/TrainedSignsPanel";
import { StatusDot } from "@/components/ui/Badge";
import { Button, TextLink } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { GlossTokens } from "@/components/ui/Gloss";
import { ModelNote } from "@/components/ui/ModelNote";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Shell } from "@/components/ui/Shell";
import { ServerUnreachable, SignInPrompt } from "@/components/ui/States";
import { Switch } from "@/components/ui/Switch";
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
  const [live, setLive] = useState<LiveSeen>({});
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
          if (message.top.length) {
            const [gloss, score] = message.top[0];
            setLive((seen) => ({ ...seen, [gloss]: { score, count: (seen[gloss]?.count ?? 0) + 1 } }));
          }
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

  const connection = {
    connecting: { tone: "warning", text: "Connecting…" },
    open: { tone: "success", text: "Connected" },
    "signed-out": { tone: "danger", text: "Signed out" },
    closed: { tone: "danger", text: "Not connected" },
  } as const;
  const linkState = connection[link];

  return (
    <Shell back="/" backLabel="Lessons" title="Translate" wide>
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 rounded-card border border-line bg-surface px-4 py-2 shadow-card">
        <Switch
          checked={debug}
          onChange={(e) => setDebug(e.target.checked)}
          label="Show debug panels"
          description="Model predictions and landmarks"
        />
        <p className="inline-flex items-center gap-2 text-sm text-muted">
          <StatusDot tone={linkState.tone} />
          Server: <span className="font-medium text-foreground">{linkState.text}</span>
        </p>
      </div>
      <div
        className={`grid gap-6 lg:items-start ${
          debug ? "lg:grid-cols-[18rem_minmax(0,1fr)_18rem]" : "lg:grid-cols-[18rem_minmax(0,1fr)]"
        }`}
      >
        <aside className="order-2 flex flex-col gap-6 lg:order-1 lg:sticky lg:top-4">
          {debug && <PredictionsPanel prediction={prediction} />}
          <TrainedSignsPanel model={model} live={live} current={prediction?.top[0]?.[0] ?? null} />
        </aside>
        <div className="order-1 flex min-w-0 flex-col gap-6 lg:order-2">
          <CameraStage
            corner={
              <StagePill>
                <StatusDot tone={linkState.tone} />
                {linkState.text}
              </StagePill>
            }
          >
            <CameraTracker onFrame={onFrame} onVideoFrame={onVideoFrame} onCamera={onCamera} />
          </CameraStage>

          {unavailable ? (
            <Callout tone="warning" role="alert">
              {unavailable}
            </Callout>
          ) : link === "signed-out" ? (
            <SignInPrompt role="alert" to="translate" />
          ) : link === "closed" ? (
            <ServerUnreachable reason="lost" />
          ) : (
            <>
              <Card as="section" aria-labelledby="signs-so-far" className="flex flex-col gap-4">
                <SectionHeading id="signs-so-far" variant="eyebrow">
                  Signs so far
                </SectionHeading>
                <p aria-live="polite" className="flex min-h-10 items-center">
                  {glosses.length ? (
                    <GlossTokens glosses={glosses} size="lg" />
                  ) : (
                    <span className="text-sm text-muted">
                      {link === "open" ? "Sign to the camera. Pause to end a sentence." : "Connecting…"}
                    </span>
                  )}
                </p>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-line pt-4">
                  <Button variant="primary" onClick={() => send("flush")} disabled={glosses.length === 0}>
                    End sentence
                  </Button>
                  <Button onClick={() => send("clear")} disabled={glosses.length === 0}>
                    Clear
                  </Button>
                  <Switch
                    checked={speak}
                    onChange={(e) => setSpeak(e.target.checked)}
                    label="Speak sentences aloud"
                    className="sm:ml-auto"
                  />
                </div>
              </Card>

              <ModelNote model={model} />

              <section aria-labelledby="sentences" className="flex flex-col gap-3">
                <SectionHeading id="sentences" variant="eyebrow">
                  Sentences
                </SectionHeading>
                {sentences.length === 0 ? (
                  <p className="text-sm text-muted">Finished sentences appear here and are saved.</p>
                ) : (
                  <ul aria-live="polite" className="flex flex-col gap-3">
                    {sentences.map((s) => (
                      <Card as="li" key={s.id} padding="sm" className="flex flex-col gap-2 sm:px-4">
                        <p className="text-lg font-medium text-foreground">{s.sentence}</p>
                        <GlossTokens glosses={s.glosses} size="sm" />
                      </Card>
                    ))}
                  </ul>
                )}
              </section>
            </>
          )}

          <p className="text-sm text-muted">
            Only landmark coordinates are sent, and only the text of each sentence is kept.{" "}
            <TextLink href="/translations">Past translations</TextLink>
            {" · "}
            <TextLink href="/translate/video">Translate a video instead</TextLink>
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
