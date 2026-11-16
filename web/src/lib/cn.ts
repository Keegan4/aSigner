/** Joins class names, skipping empty and false values. Class names stay literal for Tailwind. */
export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
