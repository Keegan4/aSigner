"use client";

import { useMemo, useState } from "react";

import type { ModelInfo, SignStat } from "@/lib/api";

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
    <th scope="col" className={`whitespace-nowrap px-1 py-1 font-medium ${align}`} aria-sort={
      sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"
    }>
      <button
        type="button"
        title={title}
        onClick={() =>
          setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : key === "label" ? 1 : -1 }))
        }
        className="underline-offset-2 hover:underline"
      >
        {label}
        {sort.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );

  return (
    <section className="flex flex-col gap-3 rounded-lg border p-4 text-sm" aria-label="Trained words">
      <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">
        Trained words{model ? ` (${words.length})` : ""}
      </h2>
      {!model ? (
        <p className="opacity-60">Waiting for the model…</p>
      ) : signs.length === 0 ? (
        <p className="opacity-60">The model did not say which words it knows.</p>
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            {model.clips != null && (
              <>
                <dt className="opacity-70">Training clips</dt>
                <dd className="text-right tabular-nums">{model.clips}</dd>
              </>
            )}
            {model.top1 != null && (
              <>
                <dt className="opacity-70">Overall accuracy</dt>
                <dd className="text-right tabular-nums">
                  {percent(model.top1)}
                  {model.test_clips ? ` of ${model.test_clips}` : ""}
                </dd>
              </>
            )}
            {counted && (
              <>
                <dt className="opacity-70">Under {FEW_CLIPS} clips</dt>
                <dd className={`text-right tabular-nums ${thin ? "text-amber-600" : ""}`}>{thin}</dd>
                <dt className="opacity-70">Never tested</dt>
                <dd className={`text-right tabular-nums ${untested ? "text-amber-600" : ""}`}>{untested}</dd>
              </>
            )}
            <dt className="opacity-70">Not seen live yet</dt>
            <dd className="text-right tabular-nums">{neverSeen}</dd>
            {model.trained_at && (
              <>
                <dt className="opacity-70">Trained</dt>
                <dd className="text-right">{new Date(model.trained_at).toLocaleDateString()}</dd>
              </>
            )}
          </dl>

          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter words"
            aria-label="Filter words"
            className="rounded-md border bg-transparent px-2 py-1"
          />

          <div className="max-h-[28rem] overflow-y-auto">
            <table className="w-full table-fixed border-collapse font-mono text-xs">
              <colgroup>
                <col />
                <col className="w-12" />
                <col className="w-12" />
                <col className="w-12" />
              </colgroup>
              <thead className="sticky top-0 bg-background">
                <tr className="border-b">
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
                      className={`border-b border-foreground/10 ${
                        s.label === current ? "bg-emerald-500/15" : ""
                      }`}
                    >
                      <td className="truncate px-1 py-1" title={name(s.label)}>
                        {name(s.label)}
                      </td>
                      <td
                        className={`px-1 py-1 text-right tabular-nums ${
                          s.clips != null && s.clips < FEW_CLIPS ? "text-amber-600" : ""
                        }`}
                      >
                        {s.clips ?? "—"}
                      </td>
                      <td
                        className={`px-1 py-1 text-right tabular-nums ${
                          s.top1 != null && s.top1 < 0.5 ? "text-red-600" : ""
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
          <p className="text-xs opacity-70">
            Test: share of held-out clips read correctly at the last training run. Live: the
            model&apos;s confidence the last time it picked this word here.
            {!counted && " Retrain the model to see per-word clips and accuracy."}
          </p>
        </>
      )}
    </section>
  );
}
