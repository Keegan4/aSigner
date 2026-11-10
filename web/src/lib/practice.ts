import { WS_URL } from "./api";
import { LEFT_HAND, POSE } from "./features";

/** Mirrors the messages of the server's /ws/practice endpoint (server/app/ws.py). */
export type PracticeResult = {
  sign_id: number;
  score: number;
  passed: boolean;
  closest: string | null;
  mastery: number;
  streak: number;
  next_review_at: string;
};

export type PracticeEvent =
  | { type: "open" }
  | { type: "ready"; sign_id: number; gloss: string }
  | ({ type: "result" } & PracticeResult)
  | { type: "error"; code: string; message: string }
  | { type: "closed"; signedOut: boolean };

export type PracticeLink = {
  start(signId: number): void;
  frame(frame: Float32Array): void;
  score(): void;
  close(): void;
};

/**
 * Set NEXT_PUBLIC_MOCK_PRACTICE=1 to try the practice screens without the
 * server or a sign model. Mock results are made up and nothing is stored.
 */
export const MOCK_PRACTICE = process.env.NEXT_PUBLIC_MOCK_PRACTICE === "1";

export function openPractice(onEvent: (event: PracticeEvent) => void): PracticeLink {
  return MOCK_PRACTICE ? openMock(onEvent) : openSocket(onEvent);
}

function openSocket(onEvent: (event: PracticeEvent) => void): PracticeLink {
  const ws = new WebSocket(`${WS_URL}/ws/practice`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => onEvent({ type: "open" });
  ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  ws.onclose = (e) => onEvent({ type: "closed", signedOut: e.code === 4401 });
  const send = (data: string | Float32Array) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(data);
  };
  return {
    start: (signId) => send(JSON.stringify({ type: "start", sign_id: signId })),
    frame: send,
    score: () => send(JSON.stringify({ type: "score" })),
    close: () => {
      ws.onclose = null;
      ws.close();
    },
  };
}

/** Passes an attempt when a hand was in view for most of it. It does not recognise signs. */
function openMock(onEvent: (event: PracticeEvent) => void): PracticeLink {
  let closed = false;
  let signId: number | null = null;
  let frames = 0;
  let withHand = 0;
  const mastery = new Map<number, number>();
  const later = (event: PracticeEvent, ms = 150) =>
    setTimeout(() => {
      if (!closed) onEvent(event);
    }, ms);
  later({ type: "open" }, 0);
  return {
    start(id) {
      signId = id;
      frames = withHand = 0;
      later({ type: "ready", sign_id: id, gloss: "" });
    },
    frame(frame) {
      if (signId === null) return;
      frames += 1;
      if (frame.subarray(LEFT_HAND, POSE).some((v) => v !== 0)) withHand += 1;
    },
    score() {
      if (signId === null) return;
      const id = signId;
      signId = null;
      const score = frames ? (0.95 * withHand) / frames : 0;
      const passed = score >= 0.6;
      const level = passed ? Math.min((mastery.get(id) ?? 0) + 1, 5) : 0;
      mastery.set(id, level);
      later(
        {
          type: "result",
          sign_id: id,
          score,
          passed,
          closest: null,
          mastery: level,
          streak: level,
          next_review_at: new Date().toISOString(),
        },
        400,
      );
    },
    close() {
      closed = true;
    },
  };
}
