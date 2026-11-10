"use client";

import { useEffect, useState } from "react";

import { getJson, type Lesson, type Progress } from "@/lib/api";

export type LessonState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "error" }
  | { status: "ready"; lesson: Lesson; progress: Progress | null };

/** Loads a lesson and, when signed in, the user's progress (null when signed out). */
export function useLesson(slug: string): LessonState {
  const [state, setState] = useState<LessonState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const res = await fetch(`/api/lessons/${encodeURIComponent(slug)}`);
      if (res.status === 404) return { status: "missing" } as const;
      if (!res.ok) throw new Error(`lesson returned ${res.status}`);
      const lesson: Lesson = await res.json();
      const progress = await getJson<Progress>("/api/progress");
      return { status: "ready", lesson, progress } as const;
    }
    load()
      .catch(() => ({ status: "error" }) as const)
      .then((next) => {
        if (!cancelled) setState(next);
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  return state;
}
