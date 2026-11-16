import type { Metadata } from "next";
import { Suspense } from "react";

import { QuizView } from "@/components/lessons/QuizView";

export const metadata: Metadata = { title: "Quiz" };

// The route's params are read on the client, so the page needs a Suspense boundary.
export default function Page() {
  return (
    <Suspense>
      <QuizView />
    </Suspense>
  );
}
