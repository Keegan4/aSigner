"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useState } from "react";

import { PracticeSession } from "@/components/PracticeSession";
import { Demo, LessonFallback, Shell } from "@/components/Shell";
import { useLesson } from "@/lib/useLesson";

export function PracticeView() {
  const { slug } = useParams<{ slug: string }>();
  const wanted = Number(useSearchParams().get("sign"));
  const state = useLesson(slug);
  const [picked, setPicked] = useState<number | null>(null);
  if (state.status !== "ready") return <LessonFallback status={state.status} />;

  const { lesson } = state;
  const back = `/lessons/${encodeURIComponent(lesson.slug)}`;
  if (lesson.signs.length === 0) {
    return (
      <Shell back={back} backLabel="Back to lesson" title={lesson.title}>
        <p className="text-sm opacity-70">This lesson has no signs yet.</p>
      </Shell>
    );
  }
  const index = Math.max(
    0,
    lesson.signs.findIndex((s) => s.id === (picked ?? wanted)),
  );
  const sign = lesson.signs[index];
  const next = lesson.signs[(index + 1) % lesson.signs.length];

  return (
    <Shell back={back} backLabel="Back to lesson" title={`Practise: ${sign.gloss}`}>
      <label className="flex items-center gap-2 text-sm">
        Sign
        <select
          value={sign.id}
          onChange={(e) => setPicked(Number(e.target.value))}
          className="rounded-md border bg-background px-2 py-1"
        >
          {lesson.signs.map((s) => (
            <option key={s.id} value={s.id}>
              {s.gloss}
            </option>
          ))}
        </select>
      </label>
      <div className="grid gap-4 sm:grid-cols-2">
        <Demo path={sign.demo_path} gloss={sign.gloss} />
        <div className="flex flex-col gap-2 text-sm">
          <p className="opacity-80">
            {sign.description ?? "Description awaiting review by an SgSL signer."}
          </p>
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
      </div>
      <PracticeSession sign={sign}>
        {lesson.signs.length > 1 && (
          <button
            onClick={() => setPicked(next.id)}
            className="self-start rounded-md border px-3 py-1.5 text-sm hover:bg-foreground/5"
          >
            Next: {next.gloss}
          </button>
        )}
      </PracticeSession>
      <Link href="/check" className="text-sm underline underline-offset-4">
        Camera not tracking? Check it here
      </Link>
    </Shell>
  );
}
