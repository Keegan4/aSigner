"use client";

import { useMemo, useState } from "react";

import { SortButton } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { SearchInput } from "@/components/ui/Field";
import { SectionHeading } from "@/components/ui/SectionHeading";
import type { ModelInfo, SignStat } from "@/lib/api";
import { cn } from "@/lib/cn";

const IDLE_LABEL = "_idle"; // server/app/recognition/features.py
/** Fewer training clips than this and a sign is marked as thinly covered (ml/train.py asks for 5). */
const FEW_CLIPS = 5;

/** Per word: the model's confidence the last time it was the top guess, and how many windows it was. */
export type LiveSeen = Record<string, { score: number; count: number }>;

type SortKey = "label" | "clips" | "top1" | "live";

const percent = (p: number) => `${Math.round(p * 100)}%`;
const name = (label: string) => (label === IDLE_LABEL ? "(no sign)" : label);

/** Unknown values sort last whichever way the column is sorted. */
function compare(a: number | null | undefined, b: number | null | undefined, dir: 1 | -1) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return (a - b) * dir;
}

/**
 * Every word the sign model was trained on, with its training clips, its accuracy
 * on held-out clips at the last training run, and the confidence it last showed
 * for that word live on this page, so a tester can see which words to trust.
 */
export function TrainedSignsPanel({
  model,
  live,
  current,
}: {
  model: ModelInfo | null;
  live: LiveSeen;
  /** The model's top guess for the newest window, highlighted in the list. */
  current: string | null;
}) {
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "label", dir: 1 });
  const signs = useMemo(() => model?.signs ?? [], [model]);

  const rows = useMemo(() => {
    const needle = filter.trim().toUpperCase();
    const shown = signs.filter((s) => !needle || name(s.label).toUpperCase().includes(needle));
    const value = (s: SignStat) =>
      sort.key === "clips" ? s.clips : sort.key === "top1" ? s.top1 : live[s.label]?.score;
    return [...shown].sort((a, b) =>
      sort.key === "label"
        ? name(a.label).localeCompare(name(b.label)) * sort.dir
        : compare(value(a), value(b), sort.dir) || name(a.label).localeCompare(name(b.label)),
    );
  }, [signs, filter, sort, live]);

  const words = signs.filter((s) => s.label !== IDLE_LABEL);
  const counted = signs.some((s) => s.clips != null);
  const thin = words.filter((s) => s.clips != null && s.clips < FEW_CLIPS).length;
  const untested = words.filter((s) => s.test_clips === 0).length;
  const neverSeen = words.filter((s) => !live[s.label]).length;

  const header = (key: SortKey, label: string, title: string, align = "text-right") => (
    <th scope="col" className={`whitespace-nowrap px-1 py-1.5 font-medium ${align}`} aria-sort={
      sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"
    }>
      <SortButton
        title={title}
        direction={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : null}
        onClick={() =>
          setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : key === "label" ? 1 : -1 }))
        }
      >
        {label}
      </SortButton>
    </th>
  );

  return (
    <Card as="section" className="flex flex-col gap-3 text-sm" aria-label="Trained words">
      <SectionHeading variant="eyebrow">Trained words{model ? ` (${words.length})` : ""}</SectionHeading>
      {!model ? (
        <p className="text-muted">Waiting for the model…</p>
      ) : signs.length === 0 ? (
        <p className="text-muted">The model did not say which words it knows.</p>
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            {model.clips != null && (
              <>
                <dt className="text-muted">Training clips</dt>
                <dd className="text-right tabular-nums">{model.clips}</dd>
              </>
            )}
            {model.top1 != null && (
              <>
                <dt className="text-muted">Overall accuracy</dt>
                <dd className="text-right tabular-nums">
                  {percent(model.top1)}
                  {model.test_clips ? ` of ${model.test_clips}` : ""}
                </dd>
              </>
            )}
            {counted && (
              <>
                <dt className="text-muted">Under {FEW_CLIPS} clips</dt>
                <dd className={cn("text-right tabular-nums", thin && "font-semibold text-warning-fg")}>{thin}</dd>
                <dt className="text-muted">Never tested</dt>
                <dd className={cn("text-right tabular-nums", untested && "font-semibold text-warning-fg")}>
                  {untested}
                </dd>
              </>
            )}
            <dt className="text-muted">Not seen live yet</dt>
            <dd className="text-right tabular-nums">{neverSeen}</dd>
            {model.trained_at && (
              <>
                <dt className="text-muted">Trained</dt>
                <dd className="text-right">{new Date(model.trained_at).toLocaleDateString()}</dd>
              </>
            )}
          </dl>

          <SearchInput
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter words"
            aria-label="Filter words"
          />

          <div className="max-h-[28rem] overflow-y-auto rounded-control border border-line">
            <table className="w-full table-fixed border-collapse font-mono text-xs">
              <colgroup>
                <col />
                <col className="w-12" />
                <col className="w-12" />
                <col className="w-12" />
              </colgroup>
              <thead className="sticky top-0 bg-sunken text-muted">
                <tr className="border-b border-line">
                  {header("label", "Word", "Sort by word", "text-left")}
                  {header("clips", "Clips", "Training clips of this word")}
                  {header("top1", "Test", "Accuracy on held-out clips at the last training run")}
                  {header("live", "Live", "Confidence the last time this was the model's top guess on this page")}
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => {
                  const seen = live[s.label];
                  return (
                    <tr
                      key={s.label}
                      className={cn(
                        "border-b border-line last:border-b-0",
                        s.label === current && "bg-success-soft font-semibold text-success-fg",
                      )}
                    >
                      <td className="truncate px-1 py-1" title={name(s.label)}>
                        {s.label === current && (
                          <>
                            <span aria-hidden>▸ </span>
                            <span className="sr-only">Current guess: </span>
                          </>
                        )}
                        {name(s.label)}
                      </td>
                      <td
                        className={`px-1 py-1 text-right tabular-nums ${
                          s.clips != null && s.clips < FEW_CLIPS ? "font-semibold text-warning-fg" : ""
                        }`}
                      >
                        {s.clips ?? "—"}
                      </td>
                      <td
                        className={`px-1 py-1 text-right tabular-nums ${
                          s.top1 != null && s.top1 < 0.5 ? "font-semibold text-danger-fg" : ""
                        }`}
                        title={
                          s.test_clips === 0
                            ? "No held-out clip of this word, so it was never tested"
                            : s.test_clips != null
                              ? `${s.test_clips} held-out clip${s.test_clips === 1 ? "" : "s"}`
                              : undefined
                        }
                      >
                        {s.top1 != null ? percent(s.top1) : s.test_clips === 0 ? "none" : "—"}
                      </td>
                      <td
                        className="px-1 py-1 text-right tabular-nums"
                        title={seen ? `Top guess in ${seen.count} window${seen.count === 1 ? "" : "s"}` : undefined}
                      >
                        {seen ? percent(seen.score) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted">
            Test: share of held-out clips read correctly at the last training run. Live: the
            model&apos;s confidence the last time it picked this word here.
            {!counted && " Retrain the model to see per-word clips and accuracy."}
          </p>
        </>
      )}
    </Card>
  );
}
