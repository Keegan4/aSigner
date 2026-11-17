import Link from "next/link";

import { IconShield } from "@/components/ui/Icons";

const LINKS = [
  { href: "/", label: "Lessons" },
  { href: "/check", label: "Camera check" },
  { href: "/train", label: "Help train the sign model" },
];

/** The footer on every page: secondary links, what aSigner is, and what it sends. */
export function AppFooter() {
  return (
    <footer className="mt-auto border-t border-line bg-surface">
      <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-10 sm:px-6 md:grid-cols-[minmax(0,1fr)_auto]">
        <div className="flex max-w-2xl flex-col gap-3 text-sm text-muted">
          <p>
            <span className="font-semibold text-foreground">aSigner is a student project.</span> It is not a
            substitute for learning from Deaf instructors, and its lesson content has not yet been reviewed
            by an SgSL signer.
          </p>
          <p className="flex gap-2">
            <IconShield size={18} className="mt-0.5 text-muted" />
            <span>
              Only landmark coordinates are sent to the server. Camera video stays in your browser and is
              never stored.
            </span>
          </p>
        </div>
        <nav aria-label="Footer">
          <ul className="flex flex-col text-sm">
            {LINKS.map((link) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className="inline-flex min-h-9 items-center rounded-control font-medium text-muted transition-colors hover:text-foreground"
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </footer>
  );
}
