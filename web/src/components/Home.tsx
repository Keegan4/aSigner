"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getJson, type LessonSummary, type Me } from "@/lib/api";

export function Home() {
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const [lessons, setLessons] = useState<LessonSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getJson<Me>("/api/me"), getJson<LessonSummary[]>("/api/lessons")])
      .then(([user, list]) => {
        setMe(user);
        setLessons(list ?? []);
      })
      .catch(() => {
        setMe(null);
        setError("The aSigner server is not reachable. Start it and reload.");
      });
  }, []);

  async function signOut() {
    await fetch("/api/auth/logout", { method: "POST" });
    setMe(null);
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-10 px-4 py-10">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">aSigner</h1>
        {me === undefined ? null : me ? (
          <div className="flex items-center gap-3 text-sm">
            <span>{me.name || me.email}</span>
            <button onClick={signOut} className="rounded-md border px-3 py-1.5 hover:bg-foreground/5">
              Sign out
            </button>
          </div>
        ) : (
          // A plain link: the server redirects the whole page to Google.
          <a
            href="/api/auth/login"
            className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background"
          >
            Sign in with Google
          </a>
        )}
      </header>

      <section className="flex flex-col gap-3">
        <p className="text-lg">
          Learn Singapore Sign Language by signing to your camera and getting feedback as you go.
        </p>
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
          aSigner is a student project. It is not a substitute for learning from Deaf instructors,
          and its lesson content has not yet been reviewed by an SgSL signer.
        </p>
      </section>

      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">Lessons</h2>
        <ul className="flex flex-col gap-2">
          {lessons.map((lesson) => (
            <li key={lesson.slug} className="rounded-lg border p-4">
              <div className="flex items-baseline justify-between gap-4">
                <span className="font-medium">{lesson.title}</span>
                <span className="text-sm opacity-60">{lesson.sign_count} signs</span>
              </div>
              <p className="mt-1 text-sm opacity-70">{lesson.summary}</p>
            </li>
          ))}
        </ul>
      </section>

      <Link href="/check" className="text-sm underline underline-offset-4">
        Check your camera and tracking
      </Link>
    </main>
  );
}
