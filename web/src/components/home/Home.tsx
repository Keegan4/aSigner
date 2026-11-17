"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { ButtonLink } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card, cardLinkCover } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { IconArrowRight, IconBook } from "@/components/ui/Icons";
import { Meter } from "@/components/ui/Meter";
import { NAV_ITEMS } from "@/components/ui/navigation";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Shell } from "@/components/ui/Shell";
import { Skeleton } from "@/components/ui/Skeleton";
import { ServerUnreachable } from "@/components/ui/States";
import { useSession } from "@/hooks/useSession";
import { getJson, type LessonSummary, type Progress } from "@/lib/api";

export function Home() {
  const { me } = useSession();
  const [lessons, setLessons] = useState<LessonSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);

  useEffect(() => {
    getJson<LessonSummary[]>("/api/lessons")
      .then((list) => setLessons(list ?? []))
      .catch(() => setError(true));
  }, []);

  // Read-only, for the per-lesson bars; without it the cards simply show no progress.
  useEffect(() => {
    if (!me) return;
    getJson<Progress>("/api/progress")
      .then((p) => setProgress(p))
      .catch(() => {});
  }, [me]);

  const first = lessons?.[0];
  const tools = NAV_ITEMS.filter((item) => item.href !== "/" && (!item.signedIn || me));

  return (
    <Shell
      hero
      width="default"
      title="Learn Singapore Sign Language"
      description="Sign to your camera and get feedback as you go."
      actions={
        first ? (
          <ButtonLink
            href={`/lessons/${encodeURIComponent(first.slug)}`}
            variant="primary"
            size="lg"
            iconAfter={<IconArrowRight size={18} />}
          >
            Start with {first.title}
          </ButtonLink>
        ) : (
          <ButtonLink href="#lessons" variant="primary" size="lg" iconAfter={<IconArrowRight size={18} />}>
            Browse lessons
          </ButtonLink>
        )
      }
    >
      <Callout tone="warning" title="A student project">
        aSigner is a student project. It is not a substitute for learning from Deaf instructors, and its
        lesson content has not yet been reviewed by an SgSL signer.
      </Callout>

      <section aria-labelledby="lessons" className="flex flex-col gap-4">
        <SectionHeading id="lessons">Lessons</SectionHeading>
        {error ? (
          <ServerUnreachable />
        ) : lessons === null ? (
          <ul aria-hidden className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1].map((i) => (
              <li key={i}>
                <Card className="flex flex-col gap-3">
                  <Skeleton className="h-6 w-1/2" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                </Card>
              </li>
            ))}
          </ul>
        ) : lessons.length === 0 ? (
          <EmptyState icon={<IconBook size={24} />} title="No lessons yet">
            The server has no lessons to show.
          </EmptyState>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {lessons.map((lesson) => (
              <li key={lesson.slug} className="flex">
                <LessonCard lesson={lesson} progress={progress} />
              </li>
            ))}
          </ul>
        )}
        {lessons === null && !error && (
          <p role="status" className="sr-only">
            Loading lessons…
          </p>
        )}
      </section>

      <section aria-labelledby="tools" className="flex flex-col gap-4">
        <SectionHeading id="tools">More in aSigner</SectionHeading>
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <li key={tool.href} className="flex">
                <Card interactive padding="sm" className="flex w-full items-start gap-3 sm:p-4">
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-control bg-accent-soft text-accent-fg">
                    <Icon size={20} />
                  </span>
                  <span className="flex min-w-0 flex-col gap-0.5">
                    <Link href={tool.href} className={`font-semibold text-foreground ${cardLinkCover}`}>
                      {tool.label}
                    </Link>
                    <span className="text-sm text-muted">{tool.description}</span>
                  </span>
                </Card>
              </li>
            );
          })}
        </ul>
      </section>
    </Shell>
  );
}

function LessonCard({ lesson, progress }: { lesson: LessonSummary; progress: Progress | null }) {
  // Progress rows carry their lesson's slug as `category`.
  const mine = progress?.signs.filter((s) => s.category === lesson.slug) ?? [];
  const mastered = progress ? mine.filter((s) => s.mastery === progress.max_mastery).length : 0;
  return (
    <Card interactive className="flex w-full flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-lg font-semibold tracking-tight">
          <Link href={`/lessons/${encodeURIComponent(lesson.slug)}`} className={cardLinkCover}>
            {lesson.title}
          </Link>
        </h3>
        <Badge className="shrink-0 tabular-nums">{lesson.sign_count} signs</Badge>
      </div>
      <p className="text-sm text-muted">{lesson.summary}</p>
      <div className="mt-auto flex flex-col gap-2">
        {progress && (
          <>
            <div className="flex items-baseline justify-between gap-2 text-xs text-muted">
              <span>
                <span className="font-semibold text-foreground tabular-nums">{mine.length}</span> of{" "}
                {lesson.sign_count} practised
              </span>
              {mastered > 0 && <span className="tabular-nums">{mastered} mastered</span>}
            </div>
            <Meter
              kind="progress"
              size="sm"
              tone="success"
              value={mine.length}
              max={Math.max(1, lesson.sign_count)}
              label={`${lesson.title}: signs practised`}
              valueText={`${mine.length} of ${lesson.sign_count}`}
            />
          </>
        )}
        <span aria-hidden className="inline-flex items-center gap-1 text-sm font-medium text-accent-fg">
          Open lesson <IconArrowRight size={16} />
        </span>
      </div>
    </Card>
  );
}
