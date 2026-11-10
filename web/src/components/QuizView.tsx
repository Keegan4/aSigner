"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

import { PracticeSession } from "@/components/PracticeSession";
import { Demo, LessonFallback, Shell } from "@/components/Shell";
import type { Lesson, Sign } from "@/lib/api";
import { useLesson } from "@/lib/useLesson";

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
  const shell = (children: React.ReactNode) => (
    <Shell back={back} backLabel="Back to lesson" title={`Quiz: ${lesson.title}`}>
      {children}
    </Shell>
  );

  if (!quiz) {
    return shell(
      <div className="grid gap-4 sm:grid-cols-2">
        <Choice
          title="Make the sign"
          text="You are shown a letter or word and sign it to the camera."
          disabled={lesson.signs.length === 0}
          onStart={() => setQuiz(makeQuiz(lesson, "produce"))}
        />
        <Choice
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
    return shell(
      <div className="flex flex-col items-start gap-4">
        <p role="status" className="text-lg">
          You got {quiz.correct} of {quiz.questions.length}.
        </p>
        <button onClick={() => setQuiz(null)} className="rounded-md border px-4 py-2 text-sm hover:bg-foreground/5">
          Another quiz
        </button>
      </div>,
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
    <button onClick={advance} className="self-start rounded-md border px-3 py-1.5 text-sm hover:bg-foreground/5">
      {quiz.at + 1 === quiz.questions.length ? "See your score" : "Next question"}
    </button>
  );

  return shell(
    <>
      <p className="text-sm opacity-70">
        Question {quiz.at + 1} of {quiz.questions.length}
      </p>
      {quiz.direction === "produce" ? (
        <>
          <p className="text-lg">
            Sign: <span className="text-3xl font-semibold">{question.sign.gloss}</span>
          </p>
          {/* Only the first attempt at each question counts. */}
          <PracticeSession sign={question.sign} onResult={(r) => answer(r.passed)}>
            {nextButton}
          </PracticeSession>
          {!quiz.answered && (
            <button onClick={advance} className="self-start text-sm underline underline-offset-4">
              Skip
            </button>
          )}
        </>
      ) : (
        <>
          <Demo path={question.sign.demo_path} gloss="this sign" />
          <div className="grid grid-cols-2 gap-3">
            {question.choices.map((choice) => {
              const right = choice.id === question.sign.id;
              const mark = !quiz.answered
                ? ""
                : right
                  ? "border-emerald-500 bg-emerald-500/10"
                  : choice.id === chosen
                    ? "border-red-500 bg-red-500/10"
                    : "opacity-60";
              return (
                <button
                  key={choice.id}
                  disabled={quiz.answered}
                  onClick={() => {
                    setChosen(choice.id);
                    answer(right);
                  }}
                  className={`rounded-lg border p-4 text-lg font-medium ${mark}`}
                >
                  {choice.gloss}
                </button>
              );
            })}
          </div>
          {quiz.answered && (
            <div role="status" className="flex flex-col gap-3">
              <p>{chosen === question.sign.id ? "Correct." : `It was ${question.sign.gloss}.`}</p>
              {nextButton}
            </div>
          )}
        </>
      )}
    </>,
  );
}

function Choice({
  title,
  text,
  note,
  disabled,
  onStart,
}: {
  title: string;
  text: string;
  note?: string;
  disabled: boolean;
  onStart: () => void;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border p-4">
      <h2 className="font-medium">{title}</h2>
      <p className="text-sm opacity-80">{text}</p>
      {note && <p className="text-sm opacity-60">{note}</p>}
      <button
        onClick={onStart}
        disabled={disabled}
        className="mt-auto self-start rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background disabled:opacity-40"
      >
        Start
      </button>
    </div>
  );
}
