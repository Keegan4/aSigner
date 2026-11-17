import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export type BadgeTone = "neutral" | "accent" | "success" | "warning" | "danger" | "info";

const TONE: Record<BadgeTone, string> = {
  neutral: "border-line bg-sunken text-muted",
  accent: "border-transparent bg-accent-soft text-accent-fg",
  success: "border-success-line bg-success-soft text-success-fg",
  warning: "border-warning-line bg-warning-soft text-warning-fg",
  danger: "border-danger-line bg-danger-soft text-danger-fg",
  info: "border-info-line bg-info-soft text-info-fg",
};

/** A small status label. Give it text or an icon with text; colour is never the only signal. */
export function Badge({
  tone = "neutral",
  icon,
  className,
  children,
}: {
  tone?: BadgeTone;
  icon?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap",
        TONE[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

/** A small coloured dot with a text label beside it, for connection and similar states. */
export function StatusDot({ tone, className }: { tone: BadgeTone; className?: string }) {
  const DOT: Record<BadgeTone, string> = {
    neutral: "bg-line-strong",
    accent: "bg-accent",
    success: "bg-success",
    warning: "bg-warning",
    danger: "bg-danger",
    info: "bg-info",
  };
  return <span aria-hidden className={cn("inline-block size-2 shrink-0 rounded-full", DOT[tone], className)} />;
}
