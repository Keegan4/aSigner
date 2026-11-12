/// <reference lib="webworker" />
/**
 * Runs some of the MediaPipe models off the page's main thread. See
 * parallel.ts, which starts one worker for the hands and one for the body.
 */
import { closeTrackers, detect, loadTrackers, type Part, type Trackers } from "./landmarkers";

export type ToWorker =
  | { type: "init"; parts: Part[]; every: Trackers["every"] }
  | { type: "frame"; image: ImageBitmap; timeMs: number }
  | { type: "close" };

export type FromWorker =
  | { type: "ready"; delegate: Trackers["delegate"] }
  | { type: "result"; result: ReturnType<typeof detect> }
  | { type: "error"; message: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;
let trackers: Trackers | undefined;

const reply = (message: FromWorker) => scope.postMessage(message);

scope.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      trackers = await loadTrackers({ parts: message.parts, every: message.every, useModule: true });
      reply({ type: "ready", delegate: trackers.delegate });
    } else if (message.type === "frame") {
      try {
        if (!trackers) throw new Error("tracker worker used before it was ready");
        reply({ type: "result", result: detect(trackers, message.image, message.timeMs) });
      } finally {
        message.image.close();
      }
    } else if (message.type === "close") {
      if (trackers) closeTrackers(trackers);
      trackers = undefined;
      scope.close();
    }
  } catch (err) {
    reply({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
