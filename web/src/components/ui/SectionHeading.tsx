import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

/**
 * The heading of a page section. `level` sets the element for the page's
 * heading order; the look stays the same. `eyebrow` is the small, uppercase
 * variant used for dense panels.
 */
export function SectionHeading({
  level = 2,
  variant = "title",
  description,
  actions,
  id,
  className,
  children,
}: {
  level?: 2 | 3;
  variant?: "title" | "eyebrow";
  description?: ReactNode;
  actions?: ReactNode;
  id?: string;
  className?: string;
  children: ReactNode;
}) {
  const Tag = level === 2 ? "h2" : "h3";
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-x-4 gap-y-2", className)}>
      <div className="flex min-w-0 flex-col gap-1">
        <Tag
          id={id}
          className={
            variant === "eyebrow"
              ? "scroll-mt-6 text-xs font-semibold tracking-wider text-muted uppercase"
              : "scroll-mt-6 text-lg font-semibold tracking-tight text-foreground"
          }
        >
          {children}
        </Tag>
        {description && <p className="text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
