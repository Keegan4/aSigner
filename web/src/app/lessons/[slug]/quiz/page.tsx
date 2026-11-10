import { Suspense } from "react";

import { QuizView } from "@/components/QuizView";

// The route's params are read on the client, so the page needs a Suspense boundary.
export default function Page() {
  return (
    <Suspense>
      <QuizView />
    </Suspense>
  );
}
