"use client";

import { useEffect, useState } from "react";

import { Shell } from "@/components/Shell";
import { getJson, type Translation } from "@/lib/api";

type State =
  | { status: "loading" | "signed-out" | "error" }
  | { status: "ready"; rows: Translation[] };

/** Reads a sentence aloud with the browser's own voice; nothing is sent anywhere. */
function speak(text: string) {
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

export function TranslationHistory() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [failed, setFailed] = useState(false);
  const [canSpeak, setCanSpeak] = useState(false);

  useEffect(() => {
    getJson<Translation[]>("/api/translations")
      .then((rows) => {
        setCanSpeak("speechSynthesis" in window);
        setState(rows ? { status: "ready", rows } : { status: "signed-out" });
      })
      .catch(() => setState({ status: "error" }));
    return () => window.speechSynthesis?.cancel();
  }, []);

  async function remove(id: number) {
    const res = await fetch(`/api/translations/${id}`, { method: "DELETE" }).catch(() => null);
    // 404 means it is already gone, which is what was asked for.
    if (!res || (!res.ok && res.status !== 404)) return setFailed(true);
    setFailed(false);
    setState((s) => (s.status === "ready" ? { ...s, rows: s.rows.filter((r) => r.id !== id) } : s));
  }

  return (
    <Shell back="/" title="Past translations">
      {state.status === "signed-out" ? (
        <p className="text-sm">
          <a href="/api/auth/login" className="underline underline-offset-4">
            Sign in with Google
          </a>{" "}
          to see your translations.
        </p>
      ) : state.status !== "ready" ? (
        <p role={state.status === "error" ? "alert" : "status"} className="text-sm opacity-70">
          {state.status === "error"
            ? "The aSigner server is not reachable. Start it and reload."
            : "Loading…"}
        </p>
      ) : (
        <>
          <p className="text-sm opacity-70">
            Only the text of a translation is kept. Deleting one removes it for good.
          </p>
          {failed && (
            <p role="alert" className="text-sm text-red-600">
              That translation could not be deleted. Try again.
            </p>
          )}
          {state.rows.length === 0 ? (
            <p className="text-sm opacity-70">No translations yet.</p>
          ) : (
            <ul className="flex flex-col gap-3">
              {state.rows.map((row) => (
                <li key={row.id} className="flex flex-col gap-2 rounded-lg border p-4">
                  <p className="text-lg">{row.sentence}</p>
                  <p className="font-mono text-sm opacity-70">{row.glosses.join(" ")}</p>
                  <div className="flex flex-wrap items-center gap-4 text-sm">
                    <time dateTime={row.created_at} className="opacity-60">
                      {new Date(row.created_at).toLocaleString()}
                    </time>
                    {canSpeak && (
                      <button onClick={() => speak(row.sentence)} className="underline underline-offset-4">
                        Speak
                      </button>
                    )}
                    <button
                      onClick={() => remove(row.id)}
                      aria-label={`Delete translation: ${row.sentence}`}
                      className="underline underline-offset-4"
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Shell>
  );
}
