export type Me = { id: number; email: string; name: string; avatar_url: string | null };
export type LessonSummary = { slug: string; title: string; summary: string; sign_count: number };
export type Sign = {
  id: number;
  gloss: string;
  description: string | null;
  demo_path: string | null;
  sign_bank_url: string | null;
};
export type Lesson = { slug: string; title: string; summary: string; signs: Sign[] };
export type SignProgress = {
  sign_id: number;
  gloss: string;
  category: string;
  mastery: number;
  streak: number;
  next_review_at: string | null;
  due: boolean;
};
export type Translation = { id: number; glosses: string[]; sentence: string; created_at: string };
export type Progress = { max_mastery: number; signs: SignProgress[] };

/**
 * WebSockets connect straight to the API server (Next rewrites only proxy HTTP).
 * The session cookie still applies because cookies are not scoped by port;
 * in production put both behind one reverse proxy and set this to its wss:// URL.
 */
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export async function getJson<T>(path: string): Promise<T | null> {
  const res = await fetch(path, { credentials: "same-origin" });
  if (res.status === 401) return null;
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

/** What the server says about the sign model (server/app/services/model_info.py). */
export type ModelInfo = {
  known: boolean;
  /** True when the model was trained or tested on too little to trust its marks. */
  limited: boolean;
  clips?: number | null;
  signers?: number | null;
  top1?: number | null;
  test_clips?: number | null;
  untested_signs?: number | null;
  trained_at?: string | null;
};
