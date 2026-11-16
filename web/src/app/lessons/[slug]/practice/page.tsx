import type { Metadata } from "next";
import { Suspense } from "react";

import { PracticeView } from "@/components/practice/PracticeView";

export const metadata: Metadata = { title: "Practise" };

// The route's params are read on the client, so the page needs a Suspense boundary.
export default function Page() {
  return (
    <Suspense>
      <PracticeView />
    </Suspense>
  );
}
