"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraStage, StagePill } from "@/components/camera/CameraStage";
import { CameraTracker } from "@/components/camera/CameraTracker";
import { StatusDot } from "@/components/ui/Badge";
import { IconEye, IconEyeOff, IconShield } from "@/components/ui/Icons";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Shell } from "@/components/ui/Shell";
import { Stat, StatGrid } from "@/components/ui/Stat";
import { WS_URL } from "@/lib/api";
import { FACE, FRAME_SIZE, LEFT_HAND, POSE, RIGHT_HAND } from "@/lib/tracking/features";

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
  const linkTone = { connecting: "warning", open: "success", "signed-out": "danger", closed: "danger" } as const;

  return (
    <Shell
      back="/"
      backLabel="Lessons"
      title="Camera check"
      width="default"
      description="Check that the camera starts and that your hands, body and face are tracked."
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:items-start">
        <CameraStage
          corner={
            <StagePill>
              <StatusDot tone={linkTone[link]} />
              {link === "open" ? "Connected" : linkText}
            </StagePill>
          }
        >
          <CameraTracker onFrame={onFrame} />
        </CameraStage>
        <div className="flex flex-col gap-4">
          <section aria-labelledby="check-seen" className="flex flex-col gap-3">
            <SectionHeading id="check-seen" variant="eyebrow">
              What the camera sees
            </SectionHeading>
            <StatGrid columns={2}>
              <Seen label="Left hand" seen={seen.left} />
              <Seen label="Right hand" seen={seen.right} />
              <Seen label="Body" seen={seen.body} />
              <Seen label="Face" seen={seen.face} />
            </StatGrid>
          </section>
          <section aria-labelledby="check-link" className="flex flex-col gap-3">
            <SectionHeading id="check-link" variant="eyebrow">
              Tracking and server
            </SectionHeading>
            <StatGrid columns={2}>
              <Stat label="Tracking rate" value={`${fps} fps`} size="sm" />
              <Stat label="Frame size" value={`${FRAME_SIZE * 4} bytes`} size="sm" />
              <Stat
                className="col-span-2"
                label="Server"
                size="sm"
                icon={<StatusDot tone={linkTone[link]} />}
                tone={link === "open" ? "success" : link === "connecting" ? "neutral" : "danger"}
                value={linkText}
              />
            </StatGrid>
          </section>
          <p className="flex gap-2 text-sm text-muted">
            <IconShield size={18} className="mt-0.5" />
            Only landmark coordinates are sent. Your video stays in this browser tab.
          </p>
        </div>
      </div>
    </Shell>
  );
}

function Seen({ label, seen }: { label: string; seen: boolean }) {
  return (
    <Stat
      label={label}
      size="sm"
      tone={seen ? "success" : "neutral"}
      value={
        <span className="inline-flex items-center gap-1.5">
          {seen ? <IconEye size={18} /> : <IconEyeOff size={18} className="text-muted" />}
          {seen ? "Seen" : "Not seen"}
        </span>
      }
    />
  );
}
