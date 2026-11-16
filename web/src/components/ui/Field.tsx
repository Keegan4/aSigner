"use client";

import {
  createContext,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
  useContext,
  useId,
} from "react";

import { IconChevronDown, IconSearch } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";

type FieldContext = { id: string; describedBy?: string; invalid: boolean };
const Context = createContext<FieldContext | null>(null);

/**
 * A labelled form control. The label, hint and error are wired to the control
 * inside it (Input, Textarea or Select) with htmlFor and aria-describedby.
 */
export function Field({
  label,
  hint,
  error,
  optional = false,
  className,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  /** Shown in place of nothing when set; marks the control invalid. */
  error?: ReactNode;
  optional?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined;
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={id} className="text-sm font-medium text-foreground">
        {label}
        {optional && <span className="font-normal text-muted"> (optional)</span>}
      </label>
      <Context.Provider value={{ id, describedBy, invalid: Boolean(error) }}>{children}</Context.Provider>
      {hint && (
        <p id={hintId} className="text-xs text-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="text-sm font-medium text-danger-fg">
          {error}
        </p>
      )}
    </div>
  );
}

const CONTROL =
  "w-full rounded-control border bg-surface px-3 text-base text-foreground transition-colors duration-150 ease-standard hover:border-foreground/60 disabled:cursor-not-allowed disabled:bg-sunken disabled:opacity-70 sm:text-sm";
const BORDER = "border-line-strong";
const INVALID = "border-danger";

function useField(id?: string, describedBy?: string) {
  const field = useContext(Context);
  return {
    id: id ?? field?.id,
    "aria-describedby": [field?.describedBy, describedBy].filter(Boolean).join(" ") || undefined,
    "aria-invalid": field?.invalid || undefined,
    invalid: Boolean(field?.invalid),
  };
}

export function Input({ id, className, "aria-describedby": describedBy, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  const { invalid, ...wired } = useField(id, describedBy);
  return <input className={cn(CONTROL, "min-h-11 pointer-fine:min-h-10", invalid ? INVALID : BORDER, className)} {...wired} {...rest} />;
}

/** A search box with a magnifier; give it an aria-label when it has no Field. */
export function SearchInput({ className, ...rest }: Omit<InputHTMLAttributes<HTMLInputElement>, "type">) {
  return (
    <div className={cn("relative", className)}>
      <IconSearch size={16} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
      <Input type="search" className="pl-9" {...rest} />
    </div>
  );
}

export function Textarea({
  id,
  className,
  "aria-describedby": describedBy,
  ...rest
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const { invalid, ...wired } = useField(id, describedBy);
  return <textarea className={cn(CONTROL, "py-2", invalid ? INVALID : BORDER, className)} {...wired} {...rest} />;
}

export function Select({
  id,
  className,
  "aria-describedby": describedBy,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement>) {
  const { invalid, ...wired } = useField(id, describedBy);
  return (
    <div className={cn("relative", className)}>
      <select
        className={cn(CONTROL, "min-h-11 appearance-none pr-9 pointer-fine:min-h-10", invalid ? INVALID : BORDER)}
        {...wired}
        {...rest}
      >
        {children}
      </select>
      <IconChevronDown size={16} className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-muted" />
    </div>
  );
}
