import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

/** What a list or page shows when there is nothing in it yet, with a next step. */
export function EmptyState({
  icon,
  title,
  children,
  actions,
  role,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  role?: "status" | "alert";
  className?: string;
}) {
  return (
    <div
      role={role}
      className={cn(
        "flex flex-col items-center gap-3 rounded-card border border-dashed border-line-strong/50 bg-surface/60 px-6 py-10 text-center",
        className,
      )}
    >
      {icon && (
        <div className="flex size-12 items-center justify-center rounded-full bg-sunken text-muted">{icon}</div>
      )}
      <p className="text-base font-semibold text-foreground">{title}</p>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
      {actions && <div className="mt-1 flex flex-wrap justify-center gap-2">{actions}</div>}
    </div>
  );
}
