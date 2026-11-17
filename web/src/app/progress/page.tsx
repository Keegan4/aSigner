import type { Metadata } from "next";

import { ProgressView } from "@/components/progress/ProgressView";

export const metadata: Metadata = { title: "Your progress" };

export default function Page() {
  return <ProgressView />;
}
