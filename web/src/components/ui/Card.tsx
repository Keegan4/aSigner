import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/cn";

type Padding = "none" | "sm" | "md" | "lg";

const PADDING: Record<Padding, string> = {
  none: "",
  sm: "p-3",
  md: "p-4 sm:p-5",
  lg: "p-5 sm:p-6",
};

/**
 * A surface with a soft border. `interactive` adds a hover state for cards that
 * hold one main link (make the link cover the card with `CardLink`).
 */
export function Card({
  as: Tag = "div",
  padding = "md",
  interactive = false,
  tone = "surface",
  className,
  children,
  ...rest
}: {
  as?: "div" | "section" | "article" | "li" | "aside" | "form";
  padding?: Padding;
  interactive?: boolean;
  /** `sunken` for a quieter panel inside a card or a page section. */
  tone?: "surface" | "sunken";
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLElement>) {
  return (
    <Tag
      className={cn(
        "rounded-card border",
        tone === "surface" ? "border-line bg-surface shadow-card" : "border-line bg-sunken",
        PADDING[padding],
        interactive &&
          "relative transition-[border-color,box-shadow] duration-150 ease-standard hover:border-line-strong/70 hover:shadow-raised focus-within:border-line-strong/70",
        className,
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}

/** Makes a card's main link cover the whole card, so the card is one big target. */
export const cardLinkCover =
  "after:absolute after:inset-0 after:rounded-card after:content-[''] focus-visible:outline-none focus-visible:after:outline-2 focus-visible:after:outline-offset-2 focus-visible:after:outline-ring";
