import type { Metadata } from "next";

import { TranslationHistory } from "@/components/translate/TranslationHistory";

export const metadata: Metadata = { title: "Past translations" };

export default function Page() {
  return <TranslationHistory />;
}
