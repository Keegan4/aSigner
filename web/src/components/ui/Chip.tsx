import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/cn";

const BASE =
  "inline-flex min-h-9 items-center gap-1.5 rounded-full border px-3 text-sm font-medium transition-colors duration-150 ease-standard pointer-fine:min-h-8 disabled:cursor-not-allowed disabled:opacity-50";
const IDLE = "border-line-strong/50 bg-surface text-foreground hover:border-line-strong hover:bg-sunken";
const SELECTED = "border-accent bg-accent text-on-accent hover:bg-accent-hover";

/**
 * A compact, selectable pill. Pass `selected` with `aria-current` or
 * `aria-pressed` (whichever the control means); the look follows `selected`.
 */
export function Chip({
  selected = false,
  dashed = false,
  icon,
  className,
  children,
  type = "button",
  ...rest
}: {
  selected?: boolean;
  /** A dashed border, for items that differ in kind (for example added on the page). */
  dashed?: boolean;
  icon?: ReactNode;
  className?: string;
  children: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type={type}
      className={cn(BASE, selected ? SELECTED : IDLE, dashed && "border-dashed", className)}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}

/** A chip that navigates. */
export function ChipLink({
  href,
  icon,
  className,
  children,
}: {
  href: string;
  icon?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Link href={href} className={cn(BASE, IDLE, className)}>
      {icon}
      {children}
    </Link>
  );
}
