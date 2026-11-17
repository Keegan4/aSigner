import type { ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { IconArrowLeft, IconArrowRight, IconCheck, IconRecord, IconTrash } from "@/components/ui/Icons";
import { Kbd } from "@/components/ui/Kbd";
import { Meter } from "@/components/ui/Meter";

/*
 * Presentational parts shared by the two take recorders (/record and /train).
 * Each page keeps its own state, timers and uploads and passes values in.
 */

/** The sign being recorded, large, with progress towards the per-sign target. */
export function CaptureSign({
  label,
  count,
  target,
  countText,
  details,
  children,
}: {
  label: string;
  count: number;
  target: number;
  /** For example "3 of 25 for this sign". */
  countText: ReactNode;
  /** Totals shown under the bar. */
  details?: ReactNode;
  /** Anything else about the sign, for example how it is made. */
  children?: ReactNode;
}) {
  const done = count >= target;
  return (
    <Card as="section" aria-label="Sign to record" className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium text-muted">Sign to record</p>
        <p className="font-mono text-6xl leading-none font-semibold tracking-tight break-all text-foreground sm:text-7xl">
          {label}
        </p>
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2 text-sm">
          <span className="font-medium text-foreground tabular-nums">{countText}</span>
          {done && (
            <span className="inline-flex items-center gap-1 font-medium text-success-fg">
              <IconCheck size={16} />
              Target reached
            </span>
          )}
        </div>
        <Meter
          kind="progress"
          tone={done ? "success" : "accent"}
          value={Math.min(count, target)}
          max={target}
          label="Takes towards the target for this sign"
          valueText={`${count} of ${target}`}
        />
        {details && <p className="text-xs text-muted">{details}</p>}
      </div>
      {children}
    </Card>
  );
}

/**
 * Previous, Record, Next and Delete last take, with the keyboard shortcuts the
 * page listens for. The key hints are visual only; the buttons keep plain names.
 */
export function CaptureControls({
  onPrevious,
  onRecord,
  onNext,
  onUndo,
  canRecord,
  canUndo,
}: {
  onPrevious: () => void;
  onRecord: () => void;
  onNext: () => void;
  onUndo: () => void;
  canRecord: boolean;
  canUndo: boolean;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] gap-2">
        <Button onClick={onPrevious} icon={<IconArrowLeft size={18} />} className="max-sm:px-3">
          <span className="max-sm:sr-only">Previous</span>
        </Button>
        <Button variant="primary" size="lg" onClick={onRecord} disabled={!canRecord} icon={<IconRecord size={20} />}>
          Record
          <Kbd className="ml-1 border-on-accent/40 bg-transparent text-on-accent shadow-none pointer-coarse:hidden">
            <span aria-hidden>Space</span>
          </Kbd>
        </Button>
        <Button onClick={onNext} iconAfter={<IconArrowRight size={18} />} className="max-sm:px-3">
          <span className="max-sm:sr-only">Next</span>
        </Button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted pointer-coarse:hidden">
          <Kbd>Space</Kbd> records · <Kbd>←</Kbd> <Kbd>→</Kbd> change the sign
        </p>
        <Button size="sm" variant="danger-ghost" onClick={onUndo} disabled={!canUndo} icon={<IconTrash size={16} />}>
          Delete last take
        </Button>
      </div>
    </div>
  );
}

export type SignChipItem = {
  key: string;
  label: string;
  /** Takes counted towards the target. */
  count: number;
  /** Shown after the count, for example everyone's takes. */
  extra?: string;
  current: boolean;
  done: boolean;
  dashed?: boolean;
  title?: string;
};

/** Every sign as a chip with its count; a tick and the word "done" (for screen readers) mark finished ones. */
export function SignChips({
  items,
  onSelect,
  empty,
}: {
  items: SignChipItem[];
  onSelect: (key: string) => void;
  empty?: ReactNode;
}) {
  return (
    <ul className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <li key={item.key}>
          <Chip
            selected={item.current}
            aria-current={item.current}
            dashed={item.dashed}
            title={item.title}
            onClick={() => onSelect(item.key)}
            className="font-mono"
          >
            {item.done && <IconCheck size={14} className={item.current ? "" : "text-success-fg"} />}
            {item.label}
            <span className={item.current ? "font-sans text-xs" : "font-sans text-xs text-muted"}>
              {" "}
              {item.count}
              {item.extra}
            </span>
            {item.done && <span className="sr-only"> (done)</span>}
          </Chip>
        </li>
      ))}
      {!items.length && empty && <li className="text-sm text-muted">{empty}</li>}
    </ul>
  );
}
