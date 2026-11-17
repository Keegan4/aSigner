import { cn } from "@/lib/cn";

const TONE = {
  accent: "bg-accent",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
  neutral: "bg-line-strong",
} as const;

/**
 * A horizontal bar for a value between 0 and `max`. `kind="meter"` for a
 * measurement (a score), `"progress"` for work towards a goal or finishing.
 * Needs a `label`; `valueText` is what a screen reader says for the value.
 */
export function Meter({
  value,
  max = 1,
  label,
  valueText,
  kind = "meter",
  tone = "accent",
  size = "md",
  className,
}: {
  value: number;
  max?: number;
  label: string;
  valueText?: string;
  kind?: "meter" | "progress";
  tone?: keyof typeof TONE;
  size?: "sm" | "md";
  className?: string;
}) {
  const share = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div
      role={kind === "meter" ? "meter" : "progressbar"}
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={Math.max(0, Math.min(max, value))}
      aria-valuetext={valueText}
      className={cn(
        "w-full overflow-hidden rounded-full bg-sunken ring-1 ring-line ring-inset",
        size === "md" ? "h-2" : "h-1.5",
        className,
      )}
    >
      <div
        className={cn("h-full rounded-full transition-[width] duration-300 ease-standard", TONE[tone])}
        style={{ width: `${share * 100}%` }}
      />
    </div>
  );
}
