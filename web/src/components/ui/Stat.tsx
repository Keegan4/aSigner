import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

const COLUMNS = {
  2: "grid-cols-2",
  3: "grid-cols-2 sm:grid-cols-3",
  "3-tight": "grid-cols-3",
  4: "grid-cols-2 sm:grid-cols-4",
} as const;

/** A row of figures: a description list, so each label stays tied to its value. */
export function StatGrid({
  columns = 3,
  className,
  children,
}: {
  columns?: keyof typeof COLUMNS;
  className?: string;
  children: ReactNode;
}) {
  return <dl className={cn("grid gap-3", COLUMNS[columns], className)}>{children}</dl>;
}

const TONE = {
  neutral: "text-foreground",
  success: "text-success-fg",
  warning: "text-warning-fg",
  danger: "text-danger-fg",
} as const;

/** One figure in a StatGrid. */
export function Stat({
  label,
  value,
  icon,
  hint,
  tone = "neutral",
  size = "md",
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  icon?: ReactNode;
  hint?: ReactNode;
  tone?: keyof typeof TONE;
  size?: "sm" | "md";
  /** For example col-span-2 to give a long value the whole row. */
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex min-w-0 flex-col gap-1 rounded-card border border-line bg-surface p-3 shadow-card sm:p-4",
        className,
      )}
    >
      <dt className="flex items-center gap-1.5 text-xs font-medium text-muted sm:text-sm">
        {icon}
        {label}
      </dt>
      <dd
        className={cn(
          "font-semibold tabular-nums",
          size === "md" ? "text-xl sm:text-2xl" : "text-sm sm:text-base",
          TONE[tone],
        )}
      >
        {value}
      </dd>
      {hint && <dd className="text-xs text-muted">{hint}</dd>}
    </div>
  );
}
