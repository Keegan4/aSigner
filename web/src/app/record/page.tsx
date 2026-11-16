import type { Metadata } from "next";

import { Recorder } from "@/components/record/Recorder";

export const metadata: Metadata = { title: "Record training clips" };

export default function Page() {
  return <Recorder />;
}
