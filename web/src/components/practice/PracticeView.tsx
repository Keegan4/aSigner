"use client";

import { useParams, useSearchParams } from "next/navigation";
import { useState } from "react";

import { PracticeSession } from "@/components/practice/PracticeSession";
import { Button, TextLink } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Field, Select } from "@/components/ui/Field";
import { IconArrowRight } from "@/components/ui/Icons";
import { Demo, LessonFallback, Shell } from "@/components/ui/Shell";
import { useLesson } from "@/hooks/useLesson";

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
        <EmptyState title="This lesson has no signs yet." />
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
    <Shell
      back={back}
      backLabel="Back to lesson"
      width="default"
      eyebrow={lesson.title}
      title={`Practise: ${sign.gloss}`}
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-start">
        <Card as="section" aria-label="Target sign" className="flex flex-col gap-4">
          <Field label="Sign">
            <Select value={sign.id} onChange={(e) => setPicked(Number(e.target.value))}>
              {lesson.signs.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.gloss}
                </option>
              ))}
            </Select>
          </Field>
          <p className="font-mono text-5xl font-semibold tracking-tight break-all text-foreground" aria-hidden>
            {sign.gloss}
          </p>
          <Demo path={sign.demo_path} gloss={sign.gloss} />
          <p className="text-sm text-muted">
            {sign.description ?? "Description awaiting review by an SgSL signer."}
          </p>
          {sign.sign_bank_url && (
            <TextLink href={sign.sign_bank_url} external className="self-start text-sm">
              SgSL Sign Bank entry
            </TextLink>
          )}
          <p className="border-t border-line pt-4 text-sm text-muted">
            Camera not tracking?{" "}
            <TextLink href="/check">Check it here</TextLink>
          </p>
        </Card>
        <PracticeSession sign={sign}>
          {lesson.signs.length > 1 && (
            <Button onClick={() => setPicked(next.id)} iconAfter={<IconArrowRight size={18} />}>
              Next: {next.gloss}
            </Button>
          )}
        </PracticeSession>
      </div>
    </Shell>
  );
}
