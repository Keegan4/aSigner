"use client";

import { useEffect, useRef, useState } from "react";

import { Button, ButtonLink } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { GlossTokens } from "@/components/ui/Gloss";
import { IconChat, IconClock, IconSpeaker, IconTrash } from "@/components/ui/Icons";
import { Shell } from "@/components/ui/Shell";
import { LoadingBlock, ServerUnreachable, SignInPrompt } from "@/components/ui/States";
import { getJson, type Translation } from "@/lib/api";

type State =
  | { status: "loading" | "signed-out" | "error" }
  | { status: "ready"; rows: Translation[] };

/** Reads a sentence aloud with the browser's own voice; nothing is sent anywhere. */
function speak(text: string) {
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

export function TranslationHistory() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [failed, setFailed] = useState(false);
  const [notice, setNotice] = useState("");
  const focusAfter = useRef<number | "main" | null>(null);
  const [canSpeak, setCanSpeak] = useState(false);

  useEffect(() => {
    getJson<Translation[]>("/api/translations")
      .then((rows) => {
        setCanSpeak("speechSynthesis" in window);
        setState(rows ? { status: "ready", rows } : { status: "signed-out" });
      })
      .catch(() => setState({ status: "error" }));
    return () => window.speechSynthesis?.cancel();
  }, []);

  async function remove(id: number) {
    const res = await fetch(`/api/translations/${id}`, { method: "DELETE" }).catch(() => null);
    // 404 means it is already gone, which is what was asked for.
    if (!res || (!res.ok && res.status !== 404)) {
      setFailed(true);
      return false;
    }
    setFailed(false);
    setState((s) => (s.status === "ready" ? { ...s, rows: s.rows.filter((r) => r.id !== id) } : s));
    return true;
  }

  /** Deletes after the row's confirmation, then says so and moves focus to the next row (or the page). */
  async function confirmDelete(id: number) {
    const rows = state.status === "ready" ? state.rows : [];
    const at = rows.findIndex((r) => r.id === id);
    const next = rows[at + 1] ?? rows[at - 1];
    const done = await remove(id);
    if (done) {
      focusAfter.current = next ? next.id : "main";
      setNotice("Translation deleted.");
    }
    return done;
  }

  // After a delete re-renders the list, focus goes where confirmDelete chose.
  useEffect(() => {
    const target = focusAfter.current;
    if (target === null) return;
    focusAfter.current = null;
    const el =
      target === "main"
        ? document.getElementById("main")
        : document.querySelector<HTMLElement>(`[data-delete-id="${target}"]`);
    el?.focus();
  }, [state]);

  return (
    <Shell
      back="/"
      backLabel="Lessons"
      title="Past translations"
      description={
        state.status === "ready"
          ? "Only the text of a translation is kept. Deleting one removes it for good."
          : undefined
      }
      actions={
        state.status === "ready" && state.rows.length > 0 ? (
          <ButtonLink href="/translate" icon={<IconChat size={18} />}>
            Translate
          </ButtonLink>
        ) : undefined
      }
    >
      {state.status === "signed-out" ? (
        <SignInPrompt to="see your translations" />
      ) : state.status === "error" ? (
        <ServerUnreachable />
      ) : state.status !== "ready" ? (
        <LoadingBlock lines={4} />
      ) : (
        <>
          <p role="status" className="sr-only">
            {notice}
          </p>
          {failed && (
            <Callout tone="danger" role="alert">
              That translation could not be deleted. Try again.
            </Callout>
          )}
          {state.rows.length === 0 ? (
            <EmptyState
              icon={<IconClock size={24} />}
              title="No translations yet."
              actions={
                <ButtonLink href="/translate" variant="primary" icon={<IconChat size={18} />}>
                  Translate
                </ButtonLink>
              }
            >
              Sentences you finish on the Translate page are kept here.
            </EmptyState>
          ) : (
            <ul className="flex flex-col gap-3">
              {state.rows.map((row) => (
                <Row key={row.id} row={row} canSpeak={canSpeak} onDelete={confirmDelete} />
              ))}
            </ul>
          )}
        </>
      )}
    </Shell>
  );
}

/** One saved translation. Delete asks once more, inline, before anything is sent. */
function Row({
  row,
  canSpeak,
  onDelete,
}: {
  row: Translation;
  canSpeak: boolean;
  /** Resolves true when the translation is gone (the row then unmounts). */
  onDelete: (id: number) => Promise<boolean>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const deleteButton = useRef<HTMLButtonElement>(null);
  const cancelButton = useRef<HTMLButtonElement>(null);
  const returnFocus = useRef(false);

  // Focus follows the question: to "Cancel" when it appears, back to "Delete" when it goes.
  useEffect(() => {
    if (confirming) cancelButton.current?.focus();
    else if (returnFocus.current) {
      returnFocus.current = false;
      deleteButton.current?.focus();
    }
  }, [confirming]);

  const cancel = () => {
    returnFocus.current = true;
    setConfirming(false);
  };

  return (
    <Card as="li" className="flex flex-col gap-3">
      <p className="text-lg font-medium text-foreground">{row.sentence}</p>
      <GlossTokens glosses={row.glosses} size="sm" />
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-line pt-3 text-sm">
        <time dateTime={row.created_at} className="inline-flex items-center gap-1.5 text-muted">
          <IconClock size={16} />
          {new Date(row.created_at).toLocaleString()}
        </time>
        {confirming ? (
          // Keyed so React mounts new buttons instead of restyling the old ones, which would fade the colours in.
          <div
            key="confirm"
            role="group"
            aria-label="Confirm delete"
            className="flex flex-wrap items-center gap-2 rounded-control bg-danger-soft py-1 pr-1 pl-3"
          >
            <span className="font-medium text-danger-fg">Delete for good?</span>
            <Button
              size="sm"
              variant="danger"
              loading={busy}
              aria-label={`Yes, delete translation: ${row.sentence}`}
              onClick={async () => {
                setBusy(true);
                if (await onDelete(row.id)) return;
                // Still here means the delete failed; the page says so above the list.
                setBusy(false);
                cancel();
              }}
            >
              Delete
            </Button>
            <Button ref={cancelButton} size="sm" variant="ghost" onClick={cancel} disabled={busy}>
              Cancel
            </Button>
          </div>
        ) : (
          <div key="actions" className="flex flex-wrap items-center gap-1">
            {canSpeak && (
              <Button size="sm" variant="ghost" icon={<IconSpeaker size={16} />} onClick={() => speak(row.sentence)}>
                Speak
              </Button>
            )}
            <Button
              ref={deleteButton}
              data-delete-id={row.id}
              size="sm"
              variant="danger-ghost"
              icon={<IconTrash size={16} />}
              onClick={() => setConfirming(true)}
              aria-label={`Delete translation: ${row.sentence}`}
            >
              Delete
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}
