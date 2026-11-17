import type { Metadata } from "next";

import { StyleGuide } from "@/components/styleguide/StyleGuide";

export const metadata: Metadata = { title: "Style guide", robots: { index: false } };

export default function Page() {
  return <StyleGuide />;
}
