import type { Metadata } from "next";

import { TrackingCheck } from "@/components/camera/TrackingCheck";

export const metadata: Metadata = { title: "Camera check" };

export default function Page() {
  return <TrackingCheck />;
}
