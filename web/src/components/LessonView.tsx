"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Demo, LessonFallback, Mastery, Shell } from "@/components/Shell";
import { useLesson } from "@/lib/useLesson";

export function LessonView() {
  const { slug } = useParams<{ slug: string }>();
  const state = useLesson(slug);
  if (state.status !== "ready") return <LessonFallback status={state.status} />;

  const { lesson, progress } = state;
  const bySign = new Map(progress?.signs.map((p) => [p.sign_id, p]));
  const base = `/lessons/${encodeURIComponent(lesson.slug)}`;

  return (
    <Shell back="/" title={lesson.title}>
      <p className="opacity-80">{lesson.summary}</p>
      <div className="flex flex-wrap gap-3 text-sm">
        <Link
          href={`${base}/practice`}
          className="rounded-md bg-foreground px-4 py-2 font-medium text-background"
        >
          Practise this lesson
        </Link>
        <Link href={`${base}/quiz`} className="rounded-md border px-4 py-2 hover:bg-foreground/5">
          Take a quiz
        </Link>
      </div>
      {!progress && (
        <p className="text-sm opacity-70">Sign in on the home page to practise and keep progress.</p>
      )}
      <ul className="grid gap-4 sm:grid-cols-2">
        {lesson.signs.map((sign) => {
          const mine = bySign.get(sign.id);
          return (
            <li key={sign.id} className="flex flex-col gap-3 rounded-lg border p-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-xl font-semibold">{sign.gloss}</h2>
                {progress && <Mastery level={mine?.mastery ?? 0} max={progress.max_mastery} />}
              </div>
              <Demo path={sign.demo_path} gloss={sign.gloss} />
              <p className="text-sm opacity-80">
                {sign.description ?? "Description awaiting review by an SgSL signer."}
              </p>
              <div className="mt-auto flex flex-wrap gap-4 text-sm">
                <Link href={`${base}/practice?sign=${sign.id}`} className="underline underline-offset-4">
                  Practise
                </Link>
                {sign.sign_bank_url && (
                  <a
                    href={sign.sign_bank_url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-4"
                  >
                    SgSL Sign Bank entry
                  </a>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </Shell>
  );
}
