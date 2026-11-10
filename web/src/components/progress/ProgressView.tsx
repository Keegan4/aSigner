"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Mastery, Shell } from "@/components/ui/Shell";
import { getJson, type LessonSummary, type Progress, type SignProgress } from "@/lib/api";

type State =
  | { status: "loading" | "signed-out" | "error" }
  | { status: "ready"; progress: Progress; lessons: LessonSummary[] };

export function ProgressView() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    Promise.all([getJson<Progress>("/api/progress"), getJson<LessonSummary[]>("/api/lessons")])
      .then(([progress, lessons]) =>
        setState(
          progress ? { status: "ready", progress, lessons: lessons ?? [] } : { status: "signed-out" },
        ),
      )
      .catch(() => setState({ status: "error" }));
  }, []);

  if (state.status !== "ready") {
    return (
      <Shell back="/" title="Your progress">
        {state.status === "signed-out" ? (
          <p className="text-sm">
            <a href="/api/auth/login" className="underline underline-offset-4">
              Sign in with Google
            </a>{" "}
            to see your progress.
          </p>
        ) : (
          <p role={state.status === "error" ? "alert" : "status"} className="text-sm opacity-70">
            {state.status === "error"
              ? "The aSigner server is not reachable. Start it and reload."
              : "Loading…"}
          </p>
        )}
      </Shell>
    );
  }

  const { progress, lessons } = state;
  const max = progress.max_mastery;
  const due = progress.signs.filter((s) => s.due);
  const mastered = progress.signs.filter((s) => s.mastery === max).length;
  const total = lessons.reduce((sum, l) => sum + l.sign_count, 0);
  const practise = (s: SignProgress) => `/lessons/${encodeURIComponent(s.category)}/practice?sign=${s.sign_id}`;

  return (
    <Shell back="/" title="Your progress">
      <dl className="grid grid-cols-3 gap-3 text-sm">
        <Stat label="Signs practised" value={`${progress.signs.length} of ${total}`} />
        <Stat label="Due for review" value={String(due.length)} />
        <Stat label="Mastered" value={String(mastered)} />
      </dl>

      {progress.signs.length === 0 && (
        <p className="text-sm opacity-70">
          Nothing practised yet. Pick a lesson on the home page to start.
        </p>
      )}

      {due.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">Review now</h2>
          <ul className="flex flex-wrap gap-2">
            {due.map((s) => (
              <li key={s.sign_id}>
                <Link href={practise(s)} className="block rounded-md border px-3 py-1.5 hover:bg-foreground/5">
                  {s.gloss}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {lessons.map((lesson) => {
        const signs = progress.signs.filter((s) => s.category === lesson.slug);
        if (signs.length === 0) return null;
        return (
          <section key={lesson.slug} className="flex flex-col gap-3">
            <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">{lesson.title}</h2>
            <table className="w-full text-left text-sm">
              <thead className="opacity-60">
                <tr>
                  <th className="py-1 font-normal">Sign</th>
                  <th className="py-1 font-normal">Mastery</th>
                  <th className="py-1 font-normal">Streak</th>
                  <th className="py-1 font-normal">Next review</th>
                </tr>
              </thead>
              <tbody>
                {signs.map((s) => (
                  <tr key={s.sign_id} className="border-t">
                    <td className="py-2 font-medium">
                      <Link href={practise(s)} className="underline underline-offset-4">
                        {s.gloss}
                      </Link>
                    </td>
                    <td className="py-2">
                      <Mastery level={s.mastery} max={max} />
                    </td>
                    <td className="py-2">{s.streak}</td>
                    <td className="py-2">
                      {s.due || !s.next_review_at
                        ? "Now"
                        : new Date(s.next_review_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        );
      })}
    </Shell>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <dt className="opacity-60">{label}</dt>
      <dd className="mt-1 text-lg font-medium">{value}</dd>
    </div>
  );
}
