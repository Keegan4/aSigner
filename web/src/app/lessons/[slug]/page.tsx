import { Suspense } from "react";

import { LessonView } from "@/components/LessonView";

// The route's params are read on the client, so the page needs a Suspense boundary.
export default function Page() {
  return (
    <Suspense>
      <LessonView />
    </Suspense>
  );
}
