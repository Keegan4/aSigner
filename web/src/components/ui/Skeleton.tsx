import { cn } from "@/lib/cn";

/** A placeholder block shown while content loads. Hidden from assistive technology. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-shimmer rounded-control bg-sunken", className)} />;
}
