import Link from "next/link";
import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ComponentProps, ReactNode } from "react";

import { IconCircleCheck, IconCircleX, IconExternal } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "danger-ghost";
export type ButtonSize = "sm" | "md" | "lg";

// Full literal class strings so Tailwind can see them.
const VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-on-accent shadow-card hover:bg-accent-hover aria-disabled:hover:bg-accent disabled:hover:bg-accent",
  secondary:
    "border border-line-strong/60 bg-surface text-foreground shadow-card hover:bg-sunken hover:border-line-strong disabled:hover:bg-surface",
  ghost: "text-foreground hover:bg-sunken disabled:hover:bg-transparent",
  danger: "bg-danger text-on-danger shadow-card hover:bg-danger-hover disabled:hover:bg-danger",
  /** A quiet destructive action that asks for confirmation before it does anything. */
  "danger-ghost": "text-danger-fg hover:bg-danger-soft disabled:hover:bg-transparent",
};

// sm stays at 36 px (32 px on fine pointers); md and lg are 44 px or more for touch.
const SIZE: Record<ButtonSize, string> = {
  sm: "min-h-9 gap-1.5 px-3 text-sm pointer-fine:min-h-8",
  md: "min-h-11 gap-2 px-4 text-sm pointer-fine:min-h-10",
  lg: "min-h-12 gap-2 px-6 text-base",
};

const BASE =
  "inline-flex shrink-0 items-center justify-center rounded-control font-medium whitespace-nowrap transition-colors duration-150 ease-standard select-none disabled:cursor-not-allowed disabled:opacity-50 aria-disabled:cursor-not-allowed aria-disabled:opacity-50";

/** The class string of a button, for the rare element that must look like one. */
export function buttonClasses({
  variant = "secondary",
  size = "md",
  block = false,
  className,
}: { variant?: ButtonVariant; size?: ButtonSize; block?: boolean; className?: string } = {}) {
  return cn(BASE, VARIANT[variant], SIZE[size], block && "w-full", className);
}

type Common = {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** An icon shown before the label (one of the Icons). */
  icon?: ReactNode;
  /** An icon shown after the label. */
  iconAfter?: ReactNode;
  /** Full width. */
  block?: boolean;
};

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  iconAfter,
  block,
  loading = false,
  type = "button",
  disabled,
  className,
  children,
  ...rest
}: Common &
  ComponentProps<"button"> & {
    /** Shows a spinner and disables the button while work is in progress. */
    loading?: boolean;
  }) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={buttonClasses({ variant, size, block, className })}
      {...rest}
    >
      {loading ? <span className="spinner size-4" aria-hidden /> : icon}
      {children}
      {iconAfter}
    </button>
  );
}

type LinkProps = Common & {
  href: string;
  children: ReactNode;
  className?: string;
} & (
    | ({
        /**
         * A plain anchor instead of next/link. Required for /api/auth/* links,
         * where the server redirects the whole page, and for other sites.
         */
        native: true;
      } & AnchorHTMLAttributes<HTMLAnchorElement>)
    | ({ native?: false } & Omit<ComponentProps<typeof Link>, "href">)
  );

/** A link that looks like a button. */
export function ButtonLink({
  href,
  native,
  variant = "secondary",
  size = "md",
  icon,
  iconAfter,
  block,
  className,
  children,
  ...rest
}: LinkProps) {
  const classes = buttonClasses({ variant, size, block, className });
  const inner = (
    <>
      {icon}
      {children}
      {iconAfter}
    </>
  );
  if (native) {
    // A full page load: the server answers /api/auth/* with a redirect.
    return (
      <a href={href} className={classes} {...(rest as AnchorHTMLAttributes<HTMLAnchorElement>)}>
        {inner}
      </a>
    );
  }
  return (
    <Link href={href} className={classes} {...(rest as Omit<ComponentProps<typeof Link>, "href">)}>
      {inner}
    </Link>
  );
}

/** An inline text link in the accent colour. Use `native` for /api/auth/* and other sites. */
export function TextLink({
  href,
  native = false,
  external = false,
  className,
  children,
  ...rest
}: {
  href: string;
  native?: boolean;
  /** Opens in a new tab and says so. Implies `native`. */
  external?: boolean;
  className?: string;
  children: ReactNode;
} & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href">) {
  const classes = cn(
    "rounded-sm font-medium text-accent-fg underline decoration-accent-fg/40 underline-offset-4 transition-colors hover:decoration-accent-fg",
    className,
  );
  if (external) {
    return (
      <a href={href} target="_blank" rel="noreferrer" className={cn(classes, "inline-flex items-center gap-1")} {...rest}>
        {children}
        <ExternalMark />
      </a>
    );
  }
  if (native) {
    return (
      <a href={href} className={classes} {...rest}>
        {children}
      </a>
    );
  }
  return (
    <Link href={href} className={classes} {...rest}>
      {children}
    </Link>
  );
}

/** The "opens in a new tab" mark for external links, with a text equivalent. */
export function ExternalMark() {
  return (
    <>
      <IconExternal size={14} />
      <span className="sr-only">(opens in a new tab)</span>
    </>
  );
}

export type OptionState = "idle" | "correct" | "wrong" | "dimmed";

const OPTION: Record<OptionState, string> = {
  idle: "border-line-strong/60 bg-surface text-foreground shadow-card hover:border-accent hover:bg-accent-soft",
  correct: "border-success bg-success-soft text-success-fg",
  wrong: "border-danger bg-danger-soft text-danger-fg",
  dimmed: "border-line bg-surface text-muted",
};

/**
 * A large answer button for a quiz. After answering, the right answer and a
 * wrong pick are marked with an icon and words as well as colour.
 */
export function OptionButton({
  state = "idle",
  className,
  children,
  ...rest
}: { state?: OptionState } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        "flex min-h-16 w-full items-center justify-between gap-3 rounded-card border-2 px-4 py-3 text-left font-mono text-xl font-semibold transition-colors duration-150 ease-standard disabled:cursor-default",
        OPTION[state],
        className,
      )}
      {...rest}
    >
      <span className="min-w-0 break-all">{children}</span>
      {state === "correct" && (
        <span className="flex shrink-0 items-center gap-1 font-sans text-sm">
          <IconCircleCheck size={22} />
          Right answer
        </span>
      )}
      {state === "wrong" && (
        <span className="flex shrink-0 items-center gap-1 font-sans text-sm">
          <IconCircleX size={22} />
          Your answer
        </span>
      )}
    </button>
  );
}

/** A column header's sort control. Put aria-sort on the <th> around it. */
export function SortButton({
  direction,
  className,
  children,
  ...rest
}: { direction: "ascending" | "descending" | null } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex min-h-7 items-center gap-0.5 rounded-sm font-medium transition-colors hover:text-foreground",
        direction ? "text-foreground" : "text-muted",
        className,
      )}
      {...rest}
    >
      {children}
      {direction === "ascending" ? " ▲" : direction === "descending" ? " ▼" : ""}
    </button>
  );
}
