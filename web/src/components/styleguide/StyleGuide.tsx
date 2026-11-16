"use client";

import { type ReactNode, useState } from "react";

import { CameraStage, StagePill, type StageOverlay } from "@/components/camera/CameraStage";
import { CaptureControls, CaptureSign, SignChips } from "@/components/camera/CapturePanel";
import { Badge, type BadgeTone, StatusDot } from "@/components/ui/Badge";
import { Button, ButtonLink, OptionButton, SortButton, TextLink } from "@/components/ui/Button";
import { Callout } from "@/components/ui/Callout";
import { Card } from "@/components/ui/Card";
import { Chip, ChipLink } from "@/components/ui/Chip";
import { EmptyState } from "@/components/ui/EmptyState";
import { Field, FileInput, Input, SearchInput, Select, Textarea } from "@/components/ui/Field";
import { GlossTokens } from "@/components/ui/Gloss";
import * as Icons from "@/components/ui/Icons";
import { Kbd } from "@/components/ui/Kbd";
import { Meter } from "@/components/ui/Meter";
import { ModelNote } from "@/components/ui/ModelNote";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { Demo, Mastery, Shell } from "@/components/ui/Shell";
import { Skeleton } from "@/components/ui/Skeleton";
import { Stat, StatGrid } from "@/components/ui/Stat";
import { LoadingBlock, ServerUnreachable, SignInPrompt } from "@/components/ui/States";
import { Switch } from "@/components/ui/Switch";

// Tokens as they are named in globals.css; each swatch reads its colour from the CSS variable.
const TOKENS: { group: string; names: string[] }[] = [
  { group: "Surfaces", names: ["background", "surface", "sunken", "stage"] },
  { group: "Text and lines", names: ["foreground", "muted", "on-stage", "line", "line-strong", "ring"] },
  { group: "Accent", names: ["accent", "accent-hover", "on-accent", "accent-soft", "accent-fg"] },
  { group: "Success", names: ["success", "success-soft", "success-line", "success-fg"] },
  { group: "Warning", names: ["warning", "warning-soft", "warning-line", "warning-fg"] },
  { group: "Danger", names: ["danger", "danger-hover", "on-danger", "danger-soft", "danger-line", "danger-fg"] },
  { group: "Info", names: ["info", "info-soft", "info-line", "info-fg"] },
];

const OVERLAYS: StageOverlay[] = [
  { kind: "none" },
  { kind: "countdown", left: 3 },
  { kind: "recording", seconds: 1 },
  { kind: "busy", label: "Checking…" },
];

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="flex flex-col gap-4 border-t border-line pt-8">
      <SectionHeading id={id}>{title}</SectionHeading>
      {children}
    </section>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-wider text-muted uppercase">{label}</h3>
      <div className="flex flex-wrap items-center gap-3">{children}</div>
    </div>
  );
}

/** Every token and shared part in its variants and states: a living reference, not linked from the app. */
export function StyleGuide() {
  const [on, setOn] = useState(true);
  const [chip, setChip] = useState("B");
  const [stage, setStage] = useState(1);
  const icons = Object.entries(Icons) as [string, (p: { size?: number }) => ReactNode][];

  return (
    <Shell
      title="Style guide"
      width="default"
      description="The design tokens and shared components every screen is built from. Colours follow the system light or dark setting."
    >
      <Section id="sg-tokens" title="Colour tokens">
        <p className="text-sm text-muted">
          Use them through Tailwind names such as <code className="font-mono">bg-surface</code>,{" "}
          <code className="font-mono">text-muted</code>, <code className="font-mono">border-line</code> and{" "}
          <code className="font-mono">text-danger-fg</code>. Palette classes such as red-600 are not used.
        </p>
        {TOKENS.map((t) => (
          <Row key={t.group} label={t.group}>
            {t.names.map((name) => (
              <div key={name} className="flex w-32 flex-col gap-1.5">
                <span
                  className="h-12 rounded-control border border-line"
                  style={{ background: `var(--${name})` }}
                  aria-hidden
                />
                <code className="font-mono text-xs text-foreground">{name}</code>
              </div>
            ))}
          </Row>
        ))}
        <Row label="Radius and elevation">
          <div className="flex size-24 items-center justify-center rounded-control border border-line bg-surface text-xs text-muted">
            control
          </div>
          <div className="flex size-24 items-center justify-center rounded-card border border-line bg-surface text-xs text-muted shadow-card">
            card
          </div>
          <div className="flex size-24 items-center justify-center rounded-card border border-line bg-surface text-xs text-muted shadow-raised">
            raised
          </div>
        </Row>
      </Section>

      <Section id="sg-type" title="Type">
        <div className="flex flex-col gap-2">
          <p className="text-4xl font-semibold tracking-tight">Page title, hero</p>
          <p className="text-3xl font-semibold tracking-tight">Page title</p>
          <p className="text-lg font-semibold tracking-tight">Section title</p>
          <p className="text-xs font-semibold tracking-wider text-muted uppercase">Panel title</p>
          <p className="text-base">Body text in Geist. Muted text sits beside it for secondary lines.</p>
          <p className="text-sm text-muted">Muted text, at least 4.5:1 on every surface.</p>
          <p className="font-mono text-sm">Glosses and data in Geist Mono: HELLO THANK-YOU 0.83</p>
          <p className="text-sm">
            <TextLink href="/styleguide">An inline link</TextLink> and an{" "}
            <TextLink href="https://example.invalid" external>
              external link
            </TextLink>
            . Keys: <Kbd>Space</Kbd> <Kbd>←</Kbd> <Kbd>→</Kbd>
          </p>
          <GlossTokens glosses={["HELLO", "NAME", "WHAT"]} size="lg" />
          <GlossTokens glosses={["HELLO", "NAME", "WHAT"]} />
          <GlossTokens glosses={["HELLO", "NAME", "WHAT"]} size="sm" />
        </div>
      </Section>

      <Section id="sg-buttons" title="Buttons">
        {(["primary", "secondary", "ghost", "danger", "danger-ghost"] as const).map((variant) => (
          <Row key={variant} label={variant}>
            <Button variant={variant} size="sm">
              Small
            </Button>
            <Button variant={variant}>Medium</Button>
            <Button variant={variant} size="lg" icon={<Icons.IconRecord size={20} />}>
              Large with icon
            </Button>
            <Button variant={variant} disabled>
              Disabled
            </Button>
            <Button variant={variant} loading>
              Loading
            </Button>
          </Row>
        ))}
        <Row label="Links styled as buttons">
          <ButtonLink href="/styleguide" variant="primary" iconAfter={<Icons.IconArrowRight size={18} />}>
            Next.js link
          </ButtonLink>
          <ButtonLink native href="#sg-buttons">
            Plain anchor (for /api/auth/*)
          </ButtonLink>
        </Row>
        <Row label="Quiz options">
          <div className="grid w-full gap-3 sm:grid-cols-2">
            <OptionButton>A</OptionButton>
            <OptionButton state="correct" disabled>
              B
            </OptionButton>
            <OptionButton state="wrong" disabled>
              C
            </OptionButton>
            <OptionButton state="dimmed" disabled>
              D
            </OptionButton>
          </div>
        </Row>
        <Row label="Sort control">
          <SortButton direction="ascending">Word</SortButton>
          <SortButton direction="descending">Clips</SortButton>
          <SortButton direction={null}>Test</SortButton>
        </Row>
      </Section>

      <Section id="sg-feedback" title="Callouts, badges and states">
        {(["info", "warning", "danger", "success", "neutral"] as const).map((tone) => (
          <Callout key={tone} tone={tone} title={`A ${tone} callout`}>
            The icon and title carry the meaning, so it never rests on colour alone.
          </Callout>
        ))}
        <Row label="Badges">
          {(["neutral", "accent", "success", "warning", "danger", "info"] as BadgeTone[]).map((tone) => (
            <Badge key={tone} tone={tone}>
              {tone}
            </Badge>
          ))}
          <Badge tone="success" icon={<Icons.IconEye size={14} />}>
            seen
          </Badge>
        </Row>
        <Row label="Status dots (always with text)">
          {(["success", "warning", "danger", "neutral"] as BadgeTone[]).map((tone) => (
            <span key={tone} className="inline-flex items-center gap-2 text-sm">
              <StatusDot tone={tone} /> {tone}
            </span>
          ))}
        </Row>
        <SignInPrompt to="see your progress" />
        <ServerUnreachable />
        <ServerUnreachable reason="lost" />
        <LoadingBlock lines={2} />
        <EmptyState
          icon={<Icons.IconClock size={24} />}
          title="Nothing here yet."
          actions={<Button variant="primary">A next step</Button>}
        >
          An empty state says what would appear here and offers a way forward.
        </EmptyState>
        <Row label="Skeleton">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-10 w-24" />
        </Row>
        <ModelNote model={{ known: true, limited: false, clips: 900, signers: 4, top1: 0.8, test_clips: 200 }} />
        <ModelNote model={{ known: true, limited: true, clips: 47, signers: 1, top1: 0.42, test_clips: 20, untested_signs: 3 }} />
      </Section>

      <Section id="sg-forms" title="Forms">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" hint="A hint under the control, read with it.">
            <Input placeholder="For example: thank you" />
          </Field>
          <Field label="Access code" error="That code was not accepted.">
            <Input defaultValue="nope" />
          </Field>
          <Field label="Frames per second">
            <Select defaultValue="15">
              <option>10</option>
              <option>15</option>
              <option>30</option>
            </Select>
          </Field>
          <Field label="Disabled">
            <Input disabled defaultValue="Not editable" />
          </Field>
          <Field label="How it is made" optional>
            <Textarea rows={2} />
          </Field>
          <Field label="Video file">
            <FileInput accept="video/*" />
          </Field>
          <SearchInput aria-label="Filter words" placeholder="Filter words" />
        </div>
        <Row label="Switch">
          <Switch label="Show debug panels" description="Model predictions and landmarks" checked={on} onChange={(e) => setOn(e.target.checked)} />
          <Switch label="Off" checked={false} onChange={() => {}} />
          <Switch label="Disabled" disabled checked onChange={() => {}} />
        </Row>
        <Row label="Chips">
          {["A", "B", "C"].map((l) => (
            <Chip key={l} selected={chip === l} aria-pressed={chip === l} onClick={() => setChip(l)}>
              {l}
            </Chip>
          ))}
          <Chip dashed>Dashed</Chip>
          <Chip disabled>Disabled</Chip>
          <ChipLink href="/styleguide" icon={<Icons.IconClock size={16} />}>
            Link chip
          </ChipLink>
        </Row>
      </Section>

      <Section id="sg-data" title="Data">
        <StatGrid>
          <Stat label="Signs practised" value="6 of 38" />
          <Stat label="Due for review" value="3" tone="warning" />
          <Stat label="Mastered" value="1" tone="success" />
        </StatGrid>
        <div className="flex max-w-md flex-col gap-3">
          <Meter value={0.83} label="Match" valueText="83%" tone="success" />
          <Meter value={0.31} label="Match" valueText="31%" tone="warning" />
          <Meter kind="progress" value={2} max={5} label="Quiz progress" />
          <Meter value={0.4} size="sm" tone="neutral" label="Probability" />
        </div>
        <Row label="Mastery">
          <Mastery level={0} max={5} />
          <Mastery level={3} max={5} />
          <Mastery level={5} max={5} />
        </Row>
        <div className="grid gap-4 sm:grid-cols-2">
          <Card>
            <p className="font-semibold">Card</p>
            <p className="text-sm text-muted">A surface with a soft border.</p>
          </Card>
          <Card tone="sunken">
            <p className="font-semibold">Sunken card</p>
            <p className="text-sm text-muted">For a quieter panel.</p>
          </Card>
        </div>
      </Section>

      <Section id="sg-camera" title="Camera and capture">
        <Row label="Overlay">
          {OVERLAYS.map((o, i) => (
            <Chip key={o.kind} selected={stage === i} aria-pressed={stage === i} onClick={() => setStage(i)}>
              {o.kind}
            </Chip>
          ))}
        </Row>
        <div className="grid gap-6 lg:grid-cols-2">
          <CameraStage
            key={stage}
            overlay={OVERLAYS[stage]}
            corner={
              <StagePill>
                <StatusDot tone="success" />
                Connected
              </StagePill>
            }
          >
            <div className="aspect-[4/3] w-full rounded-card bg-stage" />
          </CameraStage>
          <div className="flex flex-col gap-4">
            <CaptureSign label="B" count={12} target={25} countText="12 of 25 for this sign" details="40 clips in total" />
            <CaptureControls onPrevious={() => {}} onRecord={() => {}} onNext={() => {}} onUndo={() => {}} canRecord canUndo={false} />
          </div>
        </div>
        <SignChips
          items={[
            { key: "A", label: "A", count: 25, current: false, done: true },
            { key: "B", label: "B", count: 12, current: true, done: false },
            { key: "C", label: "C", count: 3, extra: " / 9", current: false, done: false },
            { key: "X", label: "COFFEE", count: 0, current: false, done: false, dashed: true },
          ]}
          onSelect={() => {}}
        />
        <div className="max-w-sm">
          <Demo path={null} gloss="A" />
        </div>
      </Section>

      <Section id="sg-icons" title="Icons">
        <p className="text-sm text-muted">Drawn on a 24 px grid in currentColor, shown at 16, 20 and 24 px.</p>
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {icons.map(([name, Icon]) => (
            <li key={name} className="flex flex-col items-center gap-2 rounded-control border border-line bg-surface p-3">
              <span className="flex items-end gap-3 text-foreground">
                <Icon size={16} />
                <Icon size={20} />
                <Icon size={24} />
              </span>
              <code className="font-mono text-xs text-muted">{name.replace(/^Icon/, "")}</code>
            </li>
          ))}
        </ul>
      </Section>
    </Shell>
  );
}
