import type { InputHTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/cn";

/**
 * An on/off control: a real checkbox with role="switch", so it is announced as
 * a switch and toggles with Space. The label is part of the target.
 */
export function Switch({
  label,
  description,
  className,
  ...rest
}: {
  label: ReactNode;
  description?: ReactNode;
  className?: string;
} & Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "role">) {
  return (
    <label
      className={cn(
        "group inline-flex min-h-11 items-center gap-3 text-sm has-disabled:cursor-not-allowed has-disabled:opacity-60 pointer-fine:min-h-9",
        !rest.disabled && "cursor-pointer",
        className,
      )}
    >
      <span className="relative inline-flex shrink-0">
        <input type="checkbox" role="switch" className="peer absolute inset-0 m-0 size-full cursor-[inherit] opacity-0" {...rest} />
        <span
          aria-hidden
          className="h-6 w-10 rounded-full border border-line-strong bg-sunken transition-colors duration-150 ease-standard peer-checked:border-accent peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-ring"
        />
        <span
          aria-hidden
          className="absolute top-1 left-1 size-4 rounded-full bg-line-strong shadow-card transition-transform duration-150 ease-standard peer-checked:translate-x-4 peer-checked:bg-on-accent"
        />
      </span>
      <span className="flex flex-col">
        <span className="font-medium text-foreground">{label}</span>
        {description && <span className="text-xs text-muted">{description}</span>}
      </span>
    </label>
  );
}
