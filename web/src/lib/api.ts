export type Me = { id: number; email: string; name: string; avatar_url: string | null };
export type LessonSummary = { slug: string; title: string; summary: string; sign_count: number };

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
