"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

import { PracticeSession } from "@/components/practice/PracticeSession";
import { Button, ButtonLink, OptionButton } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { IconArrowRight, IconFilm, IconHand, IconInfo, IconQuiz } from "@/components/ui/Icons";
import { Meter } from "@/components/ui/Meter";
import { Demo, LessonFallback, Shell } from "@/components/ui/Shell";
import type { Lesson, Sign } from "@/lib/api";
import { useLesson } from "@/hooks/useLesson";

const QUESTIONS = 5;
const CHOICES = 4;

type Direction = "recognise" | "produce";
type Question = { sign: Sign; choices: Sign[] };
type Quiz = { direction: Direction; questions: Question[]; at: number; correct: number; answered: boolean };

function shuffled<T>(items: T[]): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function makeQuiz(lesson: Lesson, direction: Direction): Quiz {
  const pool = direction === "recognise" ? lesson.signs.filter((s) => s.demo_path) : lesson.signs;
  const questions = shuffled(pool)
    .slice(0, QUESTIONS)
    .map((sign) => ({
      sign,
      choices: shuffled([
        sign,
        ...shuffled(lesson.signs.filter((s) => s.id !== sign.id)).slice(0, CHOICES - 1),
      ]),
    }));
  return { direction, questions, at: 0, correct: 0, answered: false };
}

export function QuizView() {
  const { slug } = useParams<{ slug: string }>();
  const state = useLesson(slug);
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [chosen, setChosen] = useState<number | null>(null);
  if (state.status !== "ready") return <LessonFallback status={state.status} />;

  const { lesson } = state;
  const back = `/lessons/${encodeURIComponent(lesson.slug)}`;
  const withDemo = lesson.signs.filter((s) => s.demo_path).length;
  const shell = (children: React.ReactNode, wide = false) => (
    <Shell
      back={back}
      backLabel="Back to lesson"
      title={`Quiz: ${lesson.title}`}
      width={wide ? "default" : "narrow"}
    >
      {children}
    </Shell>
  );

  if (!quiz) {
    return shell(
      <div className="grid gap-4 sm:grid-cols-2">
        <Choice
          icon={<IconHand size={22} />}
          title="Make the sign"
          text="You are shown a letter or word and sign it to the camera."
          disabled={lesson.signs.length === 0}
          onStart={() => setQuiz(makeQuiz(lesson, "produce"))}
        />
        <Choice
          icon={<IconFilm size={22} />}
          title="Name the sign"
          text="You are shown a demonstration and choose what it means."
          disabled={withDemo === 0}
          note={withDemo === 0 ? "Not available until this lesson has demonstrations." : undefined}
          onStart={() => setQuiz(makeQuiz(lesson, "recognise"))}
        />
      </div>,
    );
  }

  if (quiz.at >= quiz.questions.length) {
    const total = quiz.questions.length;
    return shell(
      <Card padding="lg" className="flex flex-col items-center gap-5 text-center">
        <span className="flex size-14 items-center justify-center rounded-full bg-accent-soft text-accent-fg">
          <IconQuiz size={28} />
        </span>
        <p role="status" className="flex flex-col gap-1">
          <span aria-hidden className="text-sm font-medium text-muted">
            Your score
          </span>
          <span aria-hidden className="text-4xl font-semibold tracking-tight tabular-nums">
            {quiz.correct} of {total}
          </span>
          <span className="sr-only">
            You got {quiz.correct} of {total}.
          </span>
        </p>
        <Meter
          kind="progress"
          tone="success"
          value={quiz.correct}
          max={Math.max(1, total)}
          label="Questions right"
          valueText={`${quiz.correct} of ${total}`}
          className="max-w-xs"
        />
        <div className="flex flex-wrap justify-center gap-2">
          <Button variant="primary" onClick={() => setQuiz(null)}>
            Another quiz
          </Button>
          <ButtonLink href={back}>Back to lesson</ButtonLink>
        </div>
      </Card>,
    );
  }

  const question = quiz.questions[quiz.at];
  const answer = (right: boolean) =>
    setQuiz((q) => q && (q.answered ? q : { ...q, answered: true, correct: q.correct + (right ? 1 : 0) }));
  const advance = () => {
    setChosen(null);
    setQuiz((q) => q && { ...q, at: q.at + 1, answered: false });
  };
  const nextButton = (
    <Button variant="primary" onClick={advance} iconAfter={<IconArrowRight size={18} />}>
      {quiz.at + 1 === quiz.questions.length ? "See your score" : "Next question"}
    </Button>
  );
  const progress = (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <p className="font-medium text-foreground">
          Question {quiz.at + 1} of {quiz.questions.length}
        </p>
        <p className="text-muted tabular-nums">{quiz.correct} right so far</p>
      </div>
      <Meter
        kind="progress"
        size="sm"
        value={quiz.at + (quiz.answered ? 1 : 0)}
        max={quiz.questions.length}
        label="Quiz progress"
        valueText={`${quiz.at + (quiz.answered ? 1 : 0)} of ${quiz.questions.length} answered`}
      />
    </div>
  );

  return shell(
    <>
      {progress}
      {quiz.direction === "produce" ? (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-start">
          <Card className="flex flex-col gap-2">
            <p className="text-sm font-medium text-muted">Sign</p>
            <p className="font-mono text-6xl font-semibold tracking-tight break-all">{question.sign.gloss}</p>
            {!quiz.answered && (
              <Button variant="ghost" size="sm" onClick={advance} className="mt-2 self-start">
                Skip
              </Button>
            )}
          </Card>
          {/* Only the first attempt at each question counts. */}
          <PracticeSession sign={question.sign} onResult={(r) => answer(r.passed)}>
            {nextButton}
          </PracticeSession>
        </div>
      ) : (
        <>
          <Demo path={question.sign.demo_path} gloss="this sign" />
          <div className="grid gap-3 sm:grid-cols-2">
            {question.choices.map((choice) => {
              const right = choice.id === question.sign.id;
              const state = !quiz.answered
                ? "idle"
                : right
                  ? "correct"
                  : choice.id === chosen
                    ? "wrong"
                    : "dimmed";
              return (
                <OptionButton
                  key={choice.id}
                  state={state}
                  disabled={quiz.answered}
                  onClick={() => {
                    setChosen(choice.id);
                    answer(right);
                  }}
                >
                  {choice.gloss}
                </OptionButton>
              );
            })}
          </div>
          {quiz.answered && (
            <div role="status" className="flex flex-col items-start gap-3">
              <p className="text-base font-medium">
                {chosen === question.sign.id ? "Correct." : `It was ${question.sign.gloss}.`}
              </p>
              {nextButton}
            </div>
          )}
        </>
      )}
    </>,
    quiz.direction === "produce",
  );
}

function Choice({
  icon,
  title,
  text,
  note,
  disabled,
  onStart,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
  note?: string;
  disabled: boolean;
  onStart: () => void;
}) {
  return (
    <Card className="flex flex-col gap-3">
      <span className="flex size-11 items-center justify-center rounded-control bg-accent-soft text-accent-fg">
        {icon}
      </span>
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      <p className="text-sm text-muted">{text}</p>
      {note && (
        <p className="flex items-start gap-2 text-sm text-muted">
          <IconInfo size={16} className="mt-0.5" />
          {note}
        </p>
      )}
      <Button variant="primary" onClick={onStart} disabled={disabled} className="mt-auto self-start">
        Start
      </Button>
    </Card>
  );
}
