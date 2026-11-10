import type { ModelInfo } from "@/lib/api";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/**
 * Says how much the sign model was trained on, so a weak model is visibly weak
 * to the learner it is marking.
 */
export function ModelNote({ model }: { model: ModelInfo | null }) {
  if (!model) return null;

  const facts: string[] = [];
  if (model.clips != null && model.signers != null) {
    facts.push(`trained on ${plural(model.clips, "clip")} from ${plural(model.signers, "signer")}`);
  }
  if (model.top1 != null && model.test_clips) {
    facts.push(
      `${Math.round(model.top1 * 100)}% correct on ${plural(model.test_clips, "held-out clip")}`,
    );
  }
  if (model.untested_signs) facts.push(`${plural(model.untested_signs, "sign")} never tested`);
  const summary = facts.length
    ? `Sign model: ${facts.join("; ")}.`
    : "This sign model does not say what it was trained on.";

  if (!model.limited) return <p className="text-sm opacity-70">{summary}</p>;
  return (
    <p role="note" className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
      {summary} That is too little to rely on: treat its results as a rough guide, not as a
      judgement of your signing.
    </p>
  );
}
