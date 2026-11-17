import {
  IconBook,
  IconCamera,
  IconChart,
  IconChat,
  IconClock,
  IconFilm,
  IconRecord,
  IconUsers,
} from "@/components/ui/Icons";

export type NavItem = {
  href: string;
  label: string;
  /** One line for the tool cards on the home page. */
  description: string;
  icon: typeof IconBook;
  /** Shown only when signed in (these pages need an account). */
  signedIn?: boolean;
  /** Kept in the header bar on large screens; the rest go under "More". */
  primary?: boolean;
};

/** Every page in the header, in order. Visibility rules match the home page links before the redesign. */
export const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Lessons",
    description: "Work through a lesson sign by sign, then practise and quiz yourself.",
    icon: IconBook,
    primary: true,
  },
  {
    href: "/progress",
    label: "Progress",
    description: "See which signs are due for review and how far you have got.",
    icon: IconChart,
    signedIn: true,
    primary: true,
  },
  {
    href: "/translate",
    label: "Translate",
    description: "Sign to the camera and see what the sign model reads, as glosses and sentences.",
    icon: IconChat,
    signedIn: true,
    primary: true,
  },
  {
    href: "/check",
    label: "Camera check",
    description: "Check that your camera works and that your hands, body and face are tracked.",
    icon: IconCamera,
    primary: true,
  },
  {
    href: "/translations",
    label: "Past translations",
    description: "Read, play aloud or delete the sentences you have translated.",
    icon: IconClock,
    signedIn: true,
  },
  {
    href: "/translate/video",
    label: "Translate a video",
    description: "Step through a recording and see what the sign model reads, window by window.",
    icon: IconFilm,
    signedIn: true,
  },
  {
    href: "/record",
    label: "Record training clips",
    description: "Record short labelled clips for training the sign model.",
    icon: IconRecord,
    signedIn: true,
  },
  {
    href: "/train",
    label: "Help train the sign model",
    description: "Record takes for the training set with an access code; no account needed.",
    icon: IconUsers,
  },
];
