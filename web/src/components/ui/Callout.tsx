import type { HTMLAttributes, ReactNode } from "react";

import { IconAlert, IconCircleCheck, IconInfo, IconWarningOctagon } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";

export type CalloutTone = "info" | "warning" | "danger" | "success" | "neutral";

const TONE: Record<CalloutTone, string> = {
  info: "border-info-line bg-info-soft",
  warning: "border-warning-line bg-warning-soft",
  danger: "border-danger-line bg-danger-soft",
  success: "border-success-line bg-success-soft",
  neutral: "border-line bg-sunken",
};

const ICON_TONE: Record<CalloutTone, string> = {
  info: "text-info-fg",
  warning: "text-warning-fg",
  danger: "text-danger-fg",
  success: "text-success-fg",
  neutral: "text-muted",
};

const ICON: Record<CalloutTone, (p: { size?: number }) => ReactNode> = {
  info: IconInfo,
  warning: IconAlert,
  danger: IconWarningOctagon,
  success: IconCircleCheck,
  neutral: IconInfo,
};

/**
 * A tinted note with an icon, so the tone never rests on colour alone. Pass
 * `role="alert"`, `role="status"` or `role="note"` through when it is one.
 */
export function Callout({
  tone = "info",
  title,
  icon,
  actions,
  as: Tag = "div",
  className,
  children,
  ...rest
}: {
  tone?: CalloutTone;
  title?: ReactNode;
  /** Replaces the tone's icon; pass `false` for none. */
  icon?: ReactNode | false;
  /** Buttons shown under the text. */
  actions?: ReactNode;
  as?: "div" | "section" | "aside";
  className?: string;
  children?: ReactNode;
} & HTMLAttributes<HTMLElement>) {
  const Icon = ICON[tone];
  const body = (
    <>
      {icon !== false && (
        <div className={cn("mt-px flex", ICON_TONE[tone])}>{icon ?? <Icon size={20} />}</div>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {title && <div className="font-semibold text-foreground">{title}</div>}
        {children && <div className="text-foreground">{children}</div>}
        {actions && <div className="mt-2 flex flex-wrap gap-2">{actions}</div>}
      </div>
    </>
  );
  return (
    <Tag
      className={cn("flex gap-3 rounded-control border px-4 py-3 text-sm leading-relaxed", TONE[tone], className)}
      {...rest}
    >
      {body}
    </Tag>
  );
}
