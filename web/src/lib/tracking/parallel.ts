/**
 * Tracks a live camera off the main thread, with the hands and the body in
 * two Web Workers that run at the same time.
 *
 * On the main thread the three models ran one after another for every frame,
 * and the page could not draw or respond while they did. Here the hands
 * worker and the body (pose and face) worker each take the newest camera
 * frame whenever they are free, so a frame costs about as long as the hand
 * model alone, and the slow face model never holds the hands up. Each hands
 * result goes out with the latest body result.
 */
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

import type { Detection, Part, Timings, Trackers } from "./landmarkers";
import type { FromWorker, ToWorker } from "./tracker.worker";

// The body worker has nothing else to do, so it runs pose on every frame it
// takes. The face model is still skipped on alternate frames: it is the
// slowest of the three, and blendshapes and head angles change slowly.
const BODY_EVERY = { pose: 1, face: 2 };

type Body = {
  pose?: NormalizedLandmark[];
  blendshapes?: number[];
  faceMatrix?: ArrayLike<number>;
};

class Lane {
  busy = false;
  private worker: Worker;
  private pending?: { resolve: (r: Detection) => void; reject: (e: Error) => void };

  private constructor(worker: Worker) {
    this.worker = worker;
  }

  static start(parts: Part[], every: Trackers["every"]): Promise<{ lane: Lane; delegate: string }> {
    const worker = new Worker(new URL("./tracker.worker.ts", import.meta.url), { type: "module" });
    const lane = new Lane(worker);
    return new Promise((resolve, reject) => {
      worker.onerror = (e) => {
        reject(new Error(e.message || "the tracking worker failed to start"));
        lane.fail(new Error(e.message || "the tracking worker stopped"));
      };
      worker.onmessage = (e: MessageEvent<FromWorker>) => {
        const message = e.data;
        if (message.type === "ready") resolve({ lane, delegate: message.delegate });
        else if (message.type === "result") lane.settle(message.result);
        else if (message.type === "error") {
          reject(new Error(message.message));
          lane.fail(new Error(message.message));
        }
      };
      lane.post({ type: "init", parts, every });
    });
  }

  private post(message: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(message, transfer);
  }

  private settle(result: Detection) {
    const pending = this.pending;
    this.pending = undefined;
    this.busy = false;
    pending?.resolve(result);
  }

  private fail(error: Error) {
    const pending = this.pending;
    this.pending = undefined;
    this.busy = false;
    pending?.reject(error);
  }

  /** Tracks one frame; the caller must check `busy` first. */
  run(image: ImageBitmap, timeMs: number): Promise<Detection> {
    this.busy = true;
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject };
      this.post({ type: "frame", image, timeMs }, [image]);
    });
  }

  close() {
    this.post({ type: "close" });
  }
}

export type ParallelResult = Detection & {
  /** Milliseconds from the camera frame being taken to its hands coming back. */
  latency: number;
};

export class ParallelTracker {
  /** Whether the models run on the GPU in both workers, or on the CPU in either. */
  readonly delegate: string;
  private hands: Lane;
  private body: Lane;
  private latestBody: Body = {};
  private bodyTimings: Timings = {};
  private closed = false;
  private handsFrame = NaN;
  private bodyFrame = NaN;

  private constructor(hands: Lane, body: Lane, delegate: string) {
    this.hands = hands;
    this.body = body;
    this.delegate = delegate;
  }

  /** Starts both workers; rejects if this browser cannot run MediaPipe in a worker. */
  static async create(): Promise<ParallelTracker> {
    if (typeof Worker === "undefined" || typeof createImageBitmap === "undefined") {
      throw new Error("this browser has no Web Workers or createImageBitmap");
    }
    const [hands, body] = await Promise.allSettled([
      Lane.start(["hands"], BODY_EVERY),
      Lane.start(["pose", "face"], BODY_EVERY),
    ]);
    if (hands.status === "rejected" || body.status === "rejected") {
      if (hands.status === "fulfilled") hands.value.lane.close();
      if (body.status === "fulfilled") body.value.lane.close();
      throw hands.status === "rejected" ? hands.reason : (body as PromiseRejectedResult).reason;
    }
    const delegate =
      hands.value.delegate === body.value.delegate ? hands.value.delegate : "GPU and CPU";
    return new ParallelTracker(hands.value.lane, body.value.lane, delegate);
  }

  /**
   * Offers the frame now showing in `video`, identified by `frame` (such as the
   * video's current time). Workers that are free and have not had this frame
   * take it; others skip it, so the same frame may be offered again until both
   * have had it. Resolves with the frame's detection once the hands worker has
   * it, or with null if the hands worker did not take it.
   */
  async offer(video: HTMLVideoElement, frame: number, timeMs: number): Promise<ParallelResult | null> {
    if (this.closed) return null;
    const takeHands = !this.hands.busy && this.handsFrame !== frame;
    const takeBody = !this.body.busy && this.bodyFrame !== frame;
    if (!takeHands && !takeBody) return null;
    if (takeHands) this.handsFrame = frame;
    if (takeBody) this.bodyFrame = frame;
    // Marked busy before the frame is grabbed, so the next offer cannot take the same lane.
    if (takeHands) this.hands.busy = true;
    if (takeBody) this.body.busy = true;
    const started = performance.now();

    if (takeBody) {
      createImageBitmap(video)
        .then((image) => this.body.run(image, timeMs))
        .then((result) => {
          this.latestBody = {
            pose: result.pose,
            blendshapes: result.blendshapes,
            faceMatrix: result.faceMatrix,
          };
          this.bodyTimings = result.timings;
        })
        .catch((err) => {
          this.body.busy = false;
          if (!this.closed) console.warn("Body tracking failed on a frame", err);
        });
    }
    if (!takeHands) return null;
    try {
      const result = await this.hands.run(await createImageBitmap(video), timeMs);
      if (this.closed) return null;
      return {
        ...result,
        ...this.latestBody,
        timings: { ...this.bodyTimings, ...result.timings },
        latency: performance.now() - started,
      };
    } catch (err) {
      this.hands.busy = false;
      if (!this.closed) console.warn("Hand tracking failed on a frame", err);
      return null;
    }
  }

  close() {
    this.closed = true;
    this.hands.close();
    this.body.close();
  }
}
