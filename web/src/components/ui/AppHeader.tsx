"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { Button, ButtonLink } from "@/components/ui/Button";
import { IconChevronDown, IconClose, IconMenu, IconSignOut } from "@/components/ui/Icons";
import { NAV_ITEMS, type NavItem } from "@/components/ui/navigation";
import { Skeleton } from "@/components/ui/Skeleton";
import { useSession } from "@/hooks/useSession";
import { cn } from "@/lib/cn";

type Item = NavItem;

/** "page" when the item is this page; "section" when this page sits under it (a lesson under Lessons). */
function match(href: string, pathname: string | null): "page" | "section" | null {
  if (!pathname) return null;
  if (pathname === href) return "page";
  if (href === "/" && pathname.startsWith("/lessons/")) return "section";
  return null;
}

export function AppHeader() {
  return (
    <header className="relative z-30 border-b border-line bg-surface">
      <a
        href="#main"
        className="sr-only rounded-control bg-accent px-4 py-2 text-sm font-medium text-on-accent focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50"
      >
        Skip to content
      </a>
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center gap-3 px-4 sm:px-6">
        <Link
          href="/"
          className="-ml-1 flex min-h-11 shrink-0 items-center gap-2.5 rounded-control px-1 text-foreground"
        >
          <Mark />
          <span className="text-lg font-semibold tracking-tight">aSigner</span>
        </Link>
        {/* The pathname is not known while a dynamic route prerenders; the fallback is the same bar without the active state. */}
        <Suspense fallback={<Nav pathname={null} />}>
          <NavHere />
        </Suspense>
      </div>
    </header>
  );
}

function Mark() {
  return (
    <svg viewBox="0 0 32 32" width={32} height={32} aria-hidden className="shrink-0">
      <rect width="32" height="32" rx="9" className="fill-accent" />
      <path
        d="M12 17.5V10a1.6 1.6 0 0 1 3.2 0v6M15.2 15.5V8.6a1.6 1.6 0 0 1 3.2 0v7M18.4 15.5v-5a1.6 1.6 0 0 1 3.2 0V18a6.5 6.5 0 0 1-6.5 6.5h-.6a6 6 0 0 1-4.8-2.4l-2.4-3.3a1.6 1.6 0 0 1 2.5-2l2.2 2.4"
        fill="none"
        strokeWidth={1.9}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="stroke-on-accent"
      />
    </svg>
  );
}

function NavHere() {
  return <Nav pathname={usePathname()} />;
}

type Open = { which: "mobile" | "more"; at: string | null } | null;

function Nav({ pathname }: { pathname: string | null }) {
  const { me, devLogin, signOut } = useSession();
  // Remembers the page a menu was opened on, so moving to another page closes it.
  const [opened, setOpened] = useState<Open>(null);
  const open = opened && opened.at === pathname ? opened.which : null;
  const moreButton = useRef<HTMLButtonElement>(null);
  const mobileButton = useRef<HTMLButtonElement>(null);
  const moreMenu = useRef<HTMLDivElement>(null);
  const mobileMenu = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const trigger = open === "more" ? moreButton : mobileButton;
    const menu = open === "more" ? moreMenu : mobileMenu;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpened(null);
      trigger.current?.focus();
    };
    const onPointer = (e: PointerEvent) => {
      const target = e.target as Node;
      if (menu.current?.contains(target) || trigger.current?.contains(target)) return;
      setOpened(null);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [open]);

  const toggle = (which: "mobile" | "more") => setOpened(open === which ? null : { which, at: pathname });
  const close = () => setOpened(null);

  const items = NAV_ITEMS.filter((item) => !item.signedIn || me);
  let primary = items.filter((item) => item.primary);
  let more = items.filter((item) => !item.primary);
  // One extra item does not need a menu of its own.
  if (more.length < 2) {
    primary = items;
    more = [];
  }
  const moreActive = more.some((item) => match(item.href, pathname));

  return (
    <>
      <nav aria-label="Main" className="hidden min-w-0 flex-1 items-center gap-1 lg:flex">
        <ul className="flex items-center gap-1">
          {primary.map((item) => (
            <li key={item.href}>
              <BarLink item={item} pathname={pathname} />
            </li>
          ))}
        </ul>
        {more.length > 0 && (
          <div className="relative">
            <button
              ref={moreButton}
              type="button"
              aria-expanded={open === "more"}
              aria-controls="nav-more"
              onClick={() => toggle("more")}
              className={cn(
                "inline-flex min-h-10 items-center gap-1 rounded-control px-3 text-sm font-medium transition-colors hover:bg-sunken hover:text-foreground",
                moreActive || open === "more" ? "text-foreground" : "text-muted",
                moreActive && "bg-sunken",
              )}
            >
              More
              <IconChevronDown size={16} className={cn("transition-transform", open === "more" && "rotate-180")} />
            </button>
            <div
              ref={moreMenu}
              id="nav-more"
              hidden={open !== "more"}
              className="absolute top-full left-0 mt-2 w-64 rounded-card border border-line bg-surface p-1.5 shadow-raised"
            >
              <ul className="flex flex-col">
                {more.map((item) => (
                  <li key={item.href}>
                    <MenuLink item={item} pathname={pathname} onNavigate={close} />
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </nav>

      <div className="ml-auto flex items-center gap-2">
        <div className="hidden items-center gap-2 lg:flex">
          <Account me={me} devLogin={devLogin} signOut={signOut} />
        </div>
        {me === null && (
          <ButtonLink native href="/api/auth/login" variant="primary" size="sm" className="lg:hidden">
            Sign in<span className="sr-only"> with Google</span>
          </ButtonLink>
        )}
        <button
          ref={mobileButton}
          type="button"
          aria-expanded={open === "mobile"}
          aria-controls="nav-mobile"
          onClick={() => toggle("mobile")}
          className="inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-control border border-line px-3 text-sm font-medium text-foreground hover:bg-sunken lg:hidden"
        >
          {open === "mobile" ? <IconClose size={20} /> : <IconMenu size={20} />}
          <span>Menu</span>
        </button>
      </div>

      <div
        ref={mobileMenu}
        id="nav-mobile"
        hidden={open !== "mobile"}
        className="absolute inset-x-0 top-full max-h-[calc(100dvh-4rem)] overflow-y-auto border-b border-line bg-surface shadow-raised lg:hidden"
      >
        <nav aria-label="Main" className="mx-auto max-w-7xl px-4 py-3 sm:px-6">
          <ul className="grid gap-0.5 sm:grid-cols-2">
            {items.map((item) => (
              <li key={item.href}>
                <MenuLink item={item} pathname={pathname} onNavigate={close} />
              </li>
            ))}
          </ul>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
            <Account me={me} devLogin={devLogin} signOut={signOut} stacked />
          </div>
        </nav>
      </div>
    </>
  );
}

function BarLink({ item, pathname }: { item: Item; pathname: string | null }) {
  const state = match(item.href, pathname);
  return (
    <Link
      href={item.href}
      aria-current={state === "page" ? "page" : undefined}
      className={cn(
        "relative inline-flex min-h-10 items-center rounded-control px-3 text-sm font-medium whitespace-nowrap transition-colors hover:bg-sunken hover:text-foreground",
        state ? "bg-sunken text-foreground" : "text-muted",
      )}
    >
      {item.label}
      {state && <span aria-hidden className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-accent" />}
    </Link>
  );
}

function MenuLink({
  item,
  pathname,
  onNavigate,
}: {
  item: Item;
  pathname: string | null;
  onNavigate: () => void;
}) {
  const state = match(item.href, pathname);
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      aria-current={state === "page" ? "page" : undefined}
      className={cn(
        "flex min-h-11 items-center gap-3 rounded-control px-3 text-sm font-medium transition-colors hover:bg-sunken",
        state ? "bg-accent-soft text-accent-fg" : "text-foreground",
      )}
    >
      <span className={state ? "text-accent-fg" : "text-muted"}>
        <Icon size={18} />
      </span>
      {item.label}
    </Link>
  );
}

function Account({
  me,
  devLogin,
  signOut,
  stacked = false,
}: {
  me: ReturnType<typeof useSession>["me"];
  devLogin: boolean;
  signOut: () => Promise<void>;
  stacked?: boolean;
}) {
  if (me === undefined) return <Skeleton className="h-9 w-40" />;
  if (me) {
    return (
      <>
        <span
          className={cn("truncate text-sm text-muted", stacked ? "mr-auto" : "max-w-48")}
          title={me.email}
        >
          {stacked ? "Signed in as " : ""}
          <span className="font-medium text-foreground">{me.name || me.email}</span>
        </span>
        <Button size="sm" variant="ghost" icon={<IconSignOut size={16} />} onClick={() => void signOut()}>
          Sign out
        </Button>
      </>
    );
  }
  // Plain links: the server redirects the whole page.
  return (
    <>
      {devLogin && (
        <ButtonLink native href="/api/auth/dev-login" size="sm">
          Local sign-in
        </ButtonLink>
      )}
      <ButtonLink native href="/api/auth/login" variant="primary" size="sm">
        Sign in with Google
      </ButtonLink>
    </>
  );
}
