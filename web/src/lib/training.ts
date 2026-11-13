/**
 * Client for the training page's API (server/app/routes/training.py).
 *
 * No sign-in: requests carry the shared access code and this browser's own random
 * contributor token, both kept in localStorage. The server stores only a hash of
 * the token, so takes can be counted and undone per browser without an account.
 */

export type TrainingSign = {
  label: string;
  description: string;
  source: "lesson" | "custom" | "idle";
  /** Takes from everyone. */
  total: number;
  /** Takes from this browser. */
  mine: number;
};

export type SavedTake = { id: number; label: string; frames: number };

/** Thrown for a refused request; `status` 403 means the access code is wrong or missing. */
export class TrainingApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const CODE_KEY = "asigner.training.code";
const TOKEN_KEY = "asigner.training.contributor";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Private windows may refuse storage; the page still works for this visit.
  }
}

function randomToken(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

let sessionToken: string | null = null;

/** This browser's contributor token, made on first use. */
export function contributorToken(): string {
  sessionToken ??= read(TOKEN_KEY);
  if (!sessionToken) {
    sessionToken = randomToken();
    write(TOKEN_KEY, sessionToken);
  }
  return sessionToken;
}

export const savedAccessCode = () => read(CODE_KEY) ?? "";
export const saveAccessCode = (code: string) => write(CODE_KEY, code || null);

async function call<T>(code: string, path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("X-Contributor", contributorToken());
  if (code) headers.set("X-Training-Code", code);
  const res = await fetch(`/api/training${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = `The server answered ${res.status}.`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status line.
    }
    throw new TrainingApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const listSigns = (code: string) => call<TrainingSign[]>(code, "/signs");

export const addSign = (code: string, label: string, description: string) =>
  call<{ label: string; description: string }>(code, "/signs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, description }),
  });

export const removeSign = (code: string, label: string) =>
  call<void>(code, `/signs/${encodeURIComponent(label)}`, { method: "DELETE" });

export const saveTake = (code: string, label: string, fps: number, frames: Float32Array) =>
  call<SavedTake>(code, `/samples?${new URLSearchParams({ label, fps: fps.toFixed(2) })}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: frames.buffer as ArrayBuffer,
  });

export const deleteTake = (code: string, id: number) =>
  call<void>(code, `/samples/${id}`, { method: "DELETE" });
