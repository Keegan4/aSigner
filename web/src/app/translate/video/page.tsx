import type { Metadata } from "next";

import { VideoTranslator } from "@/components/translate/VideoTranslator";

export const metadata: Metadata = { title: "Translate a video" };

export default function Page() {
  return <VideoTranslator />;
}
