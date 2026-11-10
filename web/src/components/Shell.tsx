import Link from "next/link";
import type { ReactNode } from "react";

/** Page frame shared by the lesson, practice, quiz and progress screens. */
export function Shell({
  back,
  backLabel = "Back",
  title,
  children,
}: {
  back: string;
  backLabel?: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-10">
      <Link href={back} className="text-sm underline underline-offset-4">
        {backLabel}
      </Link>
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {children}
    </main>
  );
}

export function LessonFallback({ status }: { status: "loading" | "missing" | "error" }) {
  const text = {
    loading: "Loading…",
    missing: "This lesson does not exist.",
    error: "The aSigner server is not reachable. Start it and reload.",
  }[status];
  return (
    <Shell back="/" title="Lesson">
      <p role={status === "loading" ? "status" : "alert"} className="text-sm opacity-70">
        {text}
      </p>
    </Shell>
  );
}

export function Mastery({ level, max }: { level: number; max: number }) {
  return (
    <span
      role="img"
      aria-label={`Mastery ${level} of ${max}`}
      className="inline-flex gap-0.5 align-middle"
    >
      {Array.from({ length: max }, (_, i) => (
        <span
          key={i}
          className={`h-2 w-2 rounded-full ${i < level ? "bg-emerald-500" : "bg-foreground/15"}`}
        />
      ))}
    </span>
  );
}

/** The demo slot. Empty until demo media that may be shown exists (see the handoff). */
export function Demo({ path, gloss }: { path: string | null; gloss: string }) {
  if (!path) {
    return (
      <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed text-sm opacity-60">
        No demonstration yet
      </div>
    );
  }
  return (
    <video
      src={path}
      aria-label={`Demonstration of ${gloss}`}
      controls
      loop
      muted
      playsInline
      className="aspect-video w-full rounded-lg bg-neutral-900"
    />
  );
}
