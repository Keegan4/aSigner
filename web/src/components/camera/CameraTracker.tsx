"use client";

import { DrawingUtils, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

import { packFrame } from "@/lib/tracking/features";
import { closeTrackers, detect, loadTrackers, type Trackers } from "@/lib/tracking/landmarkers";

type Status = "loading" | "running" | "error";

// The overlay shows shoulders, elbows and wrists only (pose landmarks 11-16).
// The pose model also guesses a few face and finger points, but they are crude
// and jump about; the hands have their own, far more precise, overlay.
const ARMS = PoseLandmarker.POSE_CONNECTIONS.filter(
  (c) => c.start >= 11 && c.start <= 16 && c.end >= 11 && c.end <= 16,
);

/**
 * Shows the mirrored camera feed with a landmark overlay and hands each
 * frame's packed features to `onFrame`. Video never leaves this component.
 */
export function CameraTracker({ onFrame }: { onFrame?: (frame: Float32Array) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onFrameRef = useRef(onFrame);
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState("Starting camera and loading models…");

  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | undefined;
    let trackers: Trackers | undefined;

    async function start() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("This browser does not allow camera access on this page.");
        }
        // Camera first so a refusal is reported before any model download.
        // Each result is assigned as it arrives so release() can always free it.
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (cancelled) return release();
        trackers = await loadTrackers();
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (cancelled || !video || !canvas) return release();
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        const draw = new DrawingUtils(ctx);
        setStatus("running");

        let lastTime = -1;
        const loop = () => {
          if (cancelled || !trackers) return;
          if (video.currentTime !== lastTime) {
            lastTime = video.currentTime;
            const result = detect(trackers, video, performance.now());
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const pose = result.pose;
            if (pose) {
              // Only draw limbs the model is confident it can see. A wrist or elbow
              // out of shot is still guessed at, and the guess jumps around.
              const sure = (i: number) => (pose[i]?.visibility ?? 1) > 0.5;
              const visible = ARMS.filter((c) => sure(c.start) && sure(c.end));
              draw.drawConnectors(pose, visible, { color: "#38bdf8", lineWidth: 2 });
            }
            for (const hand of result.hands) {
              draw.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
                color: "#f59e0b",
                lineWidth: 2,
              });
              draw.drawLandmarks(hand, { color: "#fef3c7", radius: 2 });
            }
            onFrameRef.current?.(packFrame(result, video.videoWidth, video.videoHeight));
          }
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (err) {
        release();
        if (cancelled) return;
        const denied = err instanceof DOMException && err.name === "NotAllowedError";
        const missing = err instanceof DOMException && err.name === "NotFoundError";
        setStatus("error");
        setMessage(
          denied
            ? "Camera permission was denied. Allow camera access for this site and reload."
            : missing
              ? "No camera was found on this device."
              : `Could not start tracking: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    function release() {
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      stream = undefined;
      if (trackers) closeTrackers(trackers);
      trackers = undefined;
    }

    start();
    return () => {
      cancelled = true;
      release();
    };
  }, []);

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-neutral-900">
      <video ref={videoRef} playsInline muted className="h-full w-full -scale-x-100 object-cover" />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full -scale-x-100" />
      {status !== "running" && (
        <p
          role={status === "error" ? "alert" : "status"}
          className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-neutral-200"
        >
          {message}
        </p>
      )}
    </div>
  );
}
