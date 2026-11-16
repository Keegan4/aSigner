"use client";

import type { ReactNode } from "react";

import { ButtonLink, Button } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { IconRefresh, IconSignIn } from "@/components/ui/Icons";
import { Skeleton } from "@/components/ui/Skeleton";

/**
 * "Sign in with Google to …". `to` finishes the sentence, for example
 * "see your progress". The link is a plain anchor: the server redirects the page.
 */
export function SignInPrompt({ to, role }: { to: string; role?: "alert" }) {
  return (
    <Callout
      tone="info"
      role={role}
      icon={<IconSignIn size={20} />}
      title={`Sign in with Google to ${to}.`}
      actions={
        <ButtonLink native href="/api/auth/login" variant="primary" size="md">
          Sign in with Google
        </ButtonLink>
      }
    />
  );
}

const UNREACHABLE = "The aSigner server is not reachable. Start it and reload.";
const LOST = "The connection to the aSigner server was lost. Reload to try again.";

/**
 * The server could not be reached (`reason="unreachable"`) or a live connection
 * dropped (`"lost"`), with a button that reloads the page. `message` replaces
 * the wording where a page needs to say it differently.
 */
export function ServerUnreachable({
  reason = "unreachable",
  message,
}: {
  reason?: "unreachable" | "lost";
  message?: ReactNode;
}) {
  return (
    <Callout
      tone="danger"
      role="alert"
      actions={
        <Button size="sm" icon={<IconRefresh size={16} />} onClick={() => window.location.reload()}>
          Reload
        </Button>
      }
    >
      {message ?? (reason === "lost" ? LOST : UNREACHABLE)}
    </Callout>
  );
}

/** A loading placeholder with a spoken "Loading…". `lines` sets how much space it holds. */
export function LoadingBlock({ label = "Loading…", lines = 3 }: { label?: string; lines?: number }) {
  return (
    <div role="status" className="flex flex-col gap-3">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-6 w-1/3" />
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className={i % 2 ? "h-4 w-2/3" : "h-4 w-full"} />
      ))}
    </div>
  );
}
