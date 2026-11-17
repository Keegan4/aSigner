import type { Metadata } from "next";

import { Home } from "@/components/home/Home";

export const metadata: Metadata = { title: { absolute: "aSigner: learn Singapore Sign Language" } };

export default function Page() {
  return <Home />;
}
