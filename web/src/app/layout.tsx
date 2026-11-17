import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { AppFooter } from "@/components/ui/AppFooter";
import { AppHeader } from "@/components/ui/AppHeader";
import { SessionProvider } from "@/hooks/useSession";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    template: "%s · aSigner",
    default: "aSigner",
  },
  description: "Learn Singapore Sign Language with live camera feedback",
};

export const viewport: Viewport = {
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#16161c" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <SessionProvider>
          <AppHeader />
          {children}
          <AppFooter />
        </SessionProvider>
      </body>
    </html>
  );
}
