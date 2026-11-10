"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CameraTracker } from "@/components/CameraTracker";
import { WS_URL } from "@/lib/api";
import { FACE, FRAME_SIZE, LEFT_HAND, POSE, RIGHT_HAND } from "@/lib/features";

type Link_ = "connecting" | "open" | "signed-out" | "closed";

const present = (frame: Float32Array, start: number, length: number) =>
  frame.subarray(start, start + length).some((v) => v !== 0);

export function TrackingCheck() {
  const socket = useRef<WebSocket | null>(null);
  const sent = useRef(0);
  const [link, setLink] = useState<Link_>("connecting");
  const [acked, setAcked] = useState(0);
  const [fps, setFps] = useState(0);
  const [seen, setSeen] = useState({ left: false, right: false, body: false, face: false });

  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/echo`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => setLink("open");
    ws.onmessage = (e) => setAcked(JSON.parse(e.data).frames);
    ws.onclose = (e) => setLink(e.code === 4401 ? "signed-out" : "closed");
    socket.current = ws;
    return () => ws.close();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setFps(sent.current);
      sent.current = 0;
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const onFrame = useCallback((frame: Float32Array) => {
    sent.current += 1;
    if (sent.current % 5 === 0) {
      setSeen({
        left: present(frame, LEFT_HAND, 63),
        right: present(frame, RIGHT_HAND, 63),
        body: present(frame, POSE, 75),
        face: present(frame, FACE, 52),
      });
    }
    const ws = socket.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(frame);
  }, []);

  const linkText = {
    connecting: "Connecting…",
    open: `Connected, ${acked} frames received`,
    "signed-out": "Sign in to send frames to the server",
    closed: "Not connected",
  }[link];

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-10">
      <Link href="/" className="text-sm underline underline-offset-4">
        Back
      </Link>
      <h1 className="text-2xl font-semibold tracking-tight">Camera check</h1>
      <CameraTracker onFrame={onFrame} />
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <Stat label="Tracking rate" value={`${fps} fps`} />
        <Stat label="Frame size" value={`${FRAME_SIZE * 4} bytes`} />
        <Stat label="Server" value={linkText} />
        <Stat label="Left hand" value={seen.left ? "Seen" : "Not seen"} />
        <Stat label="Right hand" value={seen.right ? "Seen" : "Not seen"} />
        <Stat label="Body / face" value={`${seen.body ? "Seen" : "Not seen"} / ${seen.face ? "Seen" : "Not seen"}`} />
      </dl>
      <p className="text-sm opacity-70">
        Only landmark coordinates are sent. Your video stays in this browser tab.
      </p>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <dt className="opacity-60">{label}</dt>
      <dd className="mt-1 font-medium">{value}</dd>
    </div>
  );
}
