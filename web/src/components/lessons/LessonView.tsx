"use client";

import { useParams } from "next/navigation";

import { ButtonLink, TextLink } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { IconHand, IconQuiz } from "@/components/ui/Icons";
import { Demo, LessonFallback, Mastery, Shell } from "@/components/ui/Shell";
import { SignInPrompt } from "@/components/ui/States";
import { useLesson } from "@/hooks/useLesson";

export function LessonView() {
  const { slug } = useParams<{ slug: string }>();
  const state = useLesson(slug);
  if (state.status !== "ready") return <LessonFallback status={state.status} />;

  const { lesson, progress } = state;
  const bySign = new Map(progress?.signs.map((p) => [p.sign_id, p]));
  const base = `/lessons/${encodeURIComponent(lesson.slug)}`;

  return (
    <Shell
      back="/"
      backLabel="Lessons"
      width="default"
      title={lesson.title}
      description={
        <>
          <p>{lesson.summary}</p>
          <p className="mt-1 text-sm">
            {lesson.signs.length} {lesson.signs.length === 1 ? "sign" : "signs"}
            {progress && ` · ${progress.signs.filter((p) => p.category === lesson.slug).length} practised`}
          </p>
        </>
      }
      actions={
        <>
          <ButtonLink href={`${base}/practice`} variant="primary" icon={<IconHand size={18} />}>
            Practise this lesson
          </ButtonLink>
          <ButtonLink href={`${base}/quiz`} icon={<IconQuiz size={18} />}>
            Take a quiz
          </ButtonLink>
        </>
      }
    >
      {!progress && <SignInPrompt to="practise and keep your progress" />}
      <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {lesson.signs.map((sign) => {
          const mine = bySign.get(sign.id);
          return (
            <Card as="li" key={sign.id} className="flex flex-col gap-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="font-mono text-2xl font-semibold tracking-tight break-all">{sign.gloss}</h2>
                {progress && <Mastery level={mine?.mastery ?? 0} max={progress.max_mastery} />}
              </div>
              <Demo path={sign.demo_path} gloss={sign.gloss} />
              <p className="text-sm text-muted">
                {sign.description ?? "Description awaiting review by an SgSL signer."}
              </p>
              <div className="mt-auto flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
                <ButtonLink href={`${base}/practice?sign=${sign.id}`} size="sm" variant="secondary">
                  Practise<span className="sr-only"> {sign.gloss}</span>
                </ButtonLink>
                {sign.sign_bank_url && (
                  <TextLink href={sign.sign_bank_url} external>
                    SgSL Sign Bank entry
                  </TextLink>
                )}
              </div>
            </Card>
          );
        })}
      </ul>
    </Shell>
  );
}
