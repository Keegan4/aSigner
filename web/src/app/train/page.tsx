import type { Metadata } from "next";

import { Trainer } from "@/components/train/Trainer";

export const metadata: Metadata = { title: "Help train the sign model" };

export default function Page() {
  return <Trainer />;
}
