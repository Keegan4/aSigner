import { cn } from "@/lib/cn";

/**
 * Glosses (the written names of signs) as a row of tokens in the mono face.
 * A space between tokens keeps them separate words for screen readers.
 */
export function GlossTokens({
  glosses,
  size = "md",
  className,
}: {
  glosses: string[];
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <span className={cn("inline-flex flex-wrap gap-1.5", className)}>
      {glosses.map((gloss, i) => (
        <span key={i} className="contents">
          {i > 0 && " "}
          <span
            className={cn(
              "rounded-md border border-line bg-sunken font-mono font-medium text-foreground",
              size === "sm" && "px-1.5 py-px text-xs",
              size === "md" && "px-2 py-0.5 text-sm",
              size === "lg" && "px-2.5 py-1 text-lg sm:text-xl",
            )}
          >
            {gloss}
          </span>
        </span>
      ))}
    </span>
  );
}
