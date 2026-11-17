"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { ButtonLink, TextLink } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ChipLink } from "@/components/ui/Chip";
import { EmptyState } from "@/components/ui/EmptyState";
import { IconChart, IconClock } from "@/components/ui/Icons";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Mastery, Shell } from "@/components/ui/Shell";
import { Stat, StatGrid } from "@/components/ui/Stat";
import { LoadingBlock, ServerUnreachable, SignInPrompt } from "@/components/ui/States";
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
      <Shell back="/" backLabel="Lessons" title="Your progress">
        {state.status === "signed-out" ? (
          <SignInPrompt to="see your progress" />
        ) : state.status === "error" ? (
          <ServerUnreachable />
        ) : (
          <LoadingBlock lines={5} />
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
    <Shell back="/" backLabel="Lessons" title="Your progress">
      <StatGrid columns="3-tight">
        <Stat label="Signs practised" value={`${progress.signs.length} of ${total}`} />
        <Stat label="Due for review" value={String(due.length)} tone={due.length ? "warning" : "neutral"} />
        <Stat label="Mastered" value={String(mastered)} tone={mastered ? "success" : "neutral"} />
      </StatGrid>

      {progress.signs.length === 0 && (
        <EmptyState
          icon={<IconChart size={24} />}
          title="Nothing practised yet."
          actions={
            <ButtonLink href="/" variant="primary">
              Browse lessons
            </ButtonLink>
          }
        >
          Pick a lesson on the home page to start.
        </EmptyState>
      )}

      {due.length > 0 && (
        <section aria-labelledby="review-now" className="flex flex-col gap-3">
          <SectionHeading
            id="review-now"
            description={`${due.length} ${due.length === 1 ? "sign is" : "signs are"} due. Each one opens its practice page.`}
          >
            Review now
          </SectionHeading>
          <ul className="flex flex-wrap gap-2">
            {due.map((s) => (
              <li key={s.sign_id}>
                <ChipLink href={practise(s)} icon={<IconClock size={16} />} className="font-mono">
                  {s.gloss}
                </ChipLink>
              </li>
            ))}
          </ul>
        </section>
      )}

      {lessons.map((lesson) => {
        const signs = progress.signs.filter((s) => s.category === lesson.slug);
        if (signs.length === 0) return null;
        const id = `lesson-${lesson.slug}`;
        return (
          <Card as="section" padding="none" key={lesson.slug} aria-labelledby={id} className="overflow-hidden">
            <div className="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-4 pb-3 sm:px-5">
              <h2 id={id} className="text-lg font-semibold tracking-tight">
                {lesson.title}
              </h2>
              <p className="text-sm text-muted tabular-nums">
                {signs.length} of {lesson.sign_count} practised
              </p>
            </div>
            <table className="w-full text-left text-sm">
              <thead className="border-y border-line bg-sunken text-xs text-muted">
                <tr>
                  <th scope="col" className="py-2 pr-2 pl-4 font-medium sm:pl-5">
                    Sign
                  </th>
                  <th scope="col" className="px-2 py-2 font-medium">
                    Mastery
                  </th>
                  <th scope="col" className="px-2 py-2 font-medium">
                    Streak
                  </th>
                  <th scope="col" className="py-2 pr-4 pl-2 font-medium sm:pr-5">
                    Next review
                  </th>
                </tr>
              </thead>
              <tbody>
                {signs.map((s) => (
                  <tr key={s.sign_id} className="border-t border-line first:border-t-0">
                    <td className="py-2.5 pr-2 pl-4 font-mono font-semibold sm:pl-5">
                      <TextLink href={practise(s)}>{s.gloss}</TextLink>
                    </td>
                    <td className="px-2 py-2.5">
                      <Mastery level={s.mastery} max={max} />
                    </td>
                    <td className="px-2 py-2.5 tabular-nums">{s.streak}</td>
                    <td className="py-2.5 pr-4 pl-2 whitespace-nowrap sm:pr-5">
                      {s.due || !s.next_review_at ? (
                        <Badge tone="warning">Now</Badge>
                      ) : (
                        <span className="tabular-nums">{new Date(s.next_review_at).toLocaleDateString()}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        );
      })}
    </Shell>
  );
}
