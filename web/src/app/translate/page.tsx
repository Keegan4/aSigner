import type { Metadata } from "next";

import { Translator } from "@/components/translate/Translator";

export const metadata: Metadata = { title: "Translate" };

export default function Page() {
  return <Translator />;
}
