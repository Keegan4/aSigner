"use client";

import { createContext, type ReactNode, useCallback, useContext, useEffect, useState } from "react";

import { getJson, type Me } from "@/lib/api";

export type Session = {
  /** undefined while loading, null when signed out or the server is not reachable. */
  me: Me | null | undefined;
  /** The server offers the localhost-only developer sign-in. */
  devLogin: boolean;
  signOut: () => Promise<void>;
};

const SessionContext = createContext<Session | null>(null);

/**
 * Asks the server once who is signed in and which sign-in methods it offers,
 * for the header and the home page.
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const [devLogin, setDevLogin] = useState(false);

  useEffect(() => {
    getJson<Me>("/api/me")
      .then((user) => setMe(user))
      .catch(() => setMe(null));
    getJson<{ dev: boolean }>("/api/auth/methods")
      .then((m) => setDevLogin(Boolean(m?.dev)))
      .catch(() => {});
  }, []);

  const signOut = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => null);
    // A full page load of the home page (not a client-side navigation), so nothing personal stays on screen.
    window.location.assign(new URL("/", window.location.href));
  }, []);

  return <SessionContext.Provider value={{ me, devLogin, signOut }}>{children}</SessionContext.Provider>;
}

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useSession needs a SessionProvider (app/layout.tsx).");
  return session;
}
