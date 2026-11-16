import { IconArrowLeft, IconFilm } from "@/components/ui/Icons";
import { EmptyState } from "@/components/ui/EmptyState";
import { ButtonLink } from "@/components/ui/Button";
import { LoadingBlock, ServerUnreachable } from "@/components/ui/States";
import { cn } from "@/lib/cn";
import Link from "next/link";
import type { ReactNode } from "react";

export type ShellWidth = "narrow" | "default" | "wide";

const WIDTH: Record<ShellWidth, string> = {
  narrow: "max-w-3xl",
  default: "max-w-6xl",
  wide: "max-w-7xl",
};

/**
 * The page frame every screen renders in: the page's one <main>, a back link,
 * the h1 with an optional description and actions, then the page itself.
 */
export function Shell({
  back,
  backLabel = "Back",
  title,
  description,
  actions,
  eyebrow,
  wide = false,
  width,
  hero = false,
  children,
}: {
  /** Where the back link goes; leave out on top-level pages. */
  back?: string;
  backLabel?: string;
  title: string;
  /** A sentence under the title. */
  description?: ReactNode;
  /** Buttons beside the title (under it on small screens). */
  actions?: ReactNode;
  /** A short label above the title. */
  eyebrow?: ReactNode;
  /** Room for side panels next to the main column on large screens. Same as width="wide". */
  wide?: boolean;
  width?: ShellWidth;
  /** A larger title, for the home page. */
  hero?: boolean;
  children: ReactNode;
}) {
  return (
    <main
      id="main"
      tabIndex={-1}
      className={cn(
        "mx-auto flex w-full flex-1 flex-col gap-6 px-4 pt-6 pb-16 outline-none sm:gap-8 sm:px-6 sm:pt-8",
        WIDTH[width ?? (wide ? "wide" : "narrow")],
      )}
    >
      <div className="flex flex-col gap-3">
        {back && (
          <Link
            href={back}
            className="-ml-1 inline-flex min-h-9 items-center gap-1.5 self-start rounded-control px-1 text-sm font-medium text-muted transition-colors hover:text-foreground"
          >
            <IconArrowLeft size={16} />
            {backLabel}
          </Link>
        )}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex min-w-0 flex-col gap-2">
            {eyebrow && <p className="text-sm font-medium text-accent-fg">{eyebrow}</p>}
            <h1
              className={cn(
                "font-semibold tracking-tight text-balance text-foreground",
                hero ? "text-3xl sm:text-4xl lg:text-5xl" : "text-2xl sm:text-3xl",
              )}
            >
              {title}
            </h1>
            {description && (
              <div className={cn("max-w-2xl text-pretty text-muted", hero ? "text-lg" : "text-base")}>
                {description}
              </div>
            )}
          </div>
          {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
        </div>
      </div>
      {children}
    </main>
  );
}

export function LessonFallback({ status }: { status: "loading" | "missing" | "error" }) {
  return (
    <Shell back="/" backLabel="Lessons" title="Lesson">
      {status === "loading" ? (
        <LoadingBlock lines={4} />
      ) : status === "missing" ? (
        <EmptyState
          role="alert"
          title="This lesson does not exist."
          actions={
            <ButtonLink href="/" variant="primary">
              See all lessons
            </ButtonLink>
          }
        />
      ) : (
        <ServerUnreachable />
      )}
    </Shell>
  );
}

export function Mastery({ level, max }: { level: number; max: number }) {
  return (
    <span
      role="img"
      aria-label={`Mastery ${level} of ${max}`}
      className="inline-flex shrink-0 items-center gap-1 align-middle"
    >
      {Array.from({ length: max }, (_, i) => (
        <span
          key={i}
          className={cn(
            "size-2.5 rounded-full",
            i < level ? "bg-success" : "border border-line-strong/70 bg-transparent",
          )}
        />
      ))}
    </span>
  );
}

/** The demo slot. Empty until demo media that may be shown exists (see the handoff). */
export function Demo({ path, gloss }: { path: string | null; gloss: string }) {
  if (!path) {
    return (
      <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded-control border border-dashed border-line-strong/50 bg-sunken text-sm text-muted">
        <IconFilm size={24} />
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
      className="aspect-video w-full rounded-control bg-stage"
    />
  );
}
