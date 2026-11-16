import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

/** A keyboard key, for shortcut hints. */
export function Kbd({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <kbd
      className={cn(
        "inline-flex min-w-6 items-center justify-center rounded-md border border-line-strong/60 bg-surface px-1.5 py-px text-xs font-medium text-foreground shadow-[inset_0_-1px_0_var(--line)]",
        className,
      )}
    >
      {children}
    </kbd>
  );
}
