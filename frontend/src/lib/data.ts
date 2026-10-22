import type { Campaign, FaultHit, StrategyAttr, TelemetryStep } from "./types";

export async function loadIndex(): Promise<string[]> {
  const res = await fetch("data/index.json");
  if (!res.ok) throw new Error(`data/index.json: ${res.status}`);
  return (await res.json()).campaigns as string[];
}

export async function loadCampaign(name: string): Promise<Campaign> {
  const res = await fetch(`data/${encodeURIComponent(name)}.json`);
  if (!res.ok) throw new Error(`data/${name}.json: ${res.status}`);
  return res.json();
}

/** Campaigns run before the upstream-localization fix and CRC audit (INC-01/02) lack the audit replica. */
export const isPreFix = (c: Campaign) => !c.strategies.some((s) => s.attr === "acc_aegis_audit");

/** Stats-row key -> strategy color/series attribute. */
export const ATTR_OF_KEY: Record<string, StrategyAttr> = {
  aegis_v2: "acc_aegis",
  aegis_audit: "acc_aegis_audit",
  ranger_aegis: "acc_ranger_aegis",
  periodic_scrubbing: "acc_periodic_scrub",
  ground_uplink: "acc_uplink",
  ranger_clipping: "acc_ranger",
  blink_recentering: "acc_blink",
  unmitigated: "acc_unmitigated",
};

/** Color for a stats row; scrub-sweep rows share the scrubbing color. */
export const rowColor = (key: string) =>
  ATTR_OF_KEY[key] ? `var(--s-${ATTR_OF_KEY[key]})` : "var(--s-acc_periodic_scrub)";

export const archLabel = (arch: string) => (arch === "mobilenet_v2" ? "MobileNetV2" : "ResNet-18");

export const seedKeys = (c: Campaign) => Object.keys(c.seeds).sort((a, b) => Number(a) - Number(b));

/** Accuracy series for one strategy: one seed, or the mean (with min/max band) across seeds. */
export function accuracySeries(c: Campaign, attr: StrategyAttr, seed: string | "mean") {
  const runs = seed === "mean" ? seedKeys(c).map((k) => c.seeds[k]) : [c.seeds[seed]];
  const steps = runs[0].telemetry.length;
  return Array.from({ length: steps }, (_, i) => {
    const v = runs.map((r) => r.telemetry[i].acc[attr] ?? NaN);
    return { i, mean: v.reduce((a, b) => a + b, 0) / v.length, lo: Math.min(...v), hi: Math.max(...v) };
  });
}

export type Severity = "critical" | "moderate" | "minor";

/** How much a single hit changes the stored value (sign flips on variances are fatal: sqrt(<0) = NaN). */
export function severity(h: FaultHit): Severity {
  if (typeof h.to === "string") return "critical";
  if (h.p.endsWith("running_var") && h.to < 0) return "critical";
  const ratio = Math.abs(h.to) / Math.max(Math.abs(h.from), 1e-12);
  if (h.bit === 30 || ratio > 1e3) return "critical";
  if (ratio > 3 || ratio < 1 / 3) return "moderate";
  return "minor";
}

/** Strategies whose per-step repair actions are recorded, so their memory state can be replayed. */
export const MEMORY_STRATEGIES: { attr: StrategyAttr; label: string }[] = [
  { attr: "acc_unmitigated", label: "Unmitigated" },
  { attr: "acc_periodic_scrub", label: "Scrubbing (12 h)" },
  { attr: "acc_aegis", label: "Aegis-v2" },
];

export interface BlockState {
  hits: { h: FaultHit; step: number; sev: Severity }[]; // hits since the block was last clean
  restoredAt: number | null; // most recent restore step <= current
}

/** Corruption state of every block at `step` (inclusive) for a strategy with recorded repairs. */
export function memoryState(
  telemetry: TelemetryStep[],
  faults: FaultHit[][],
  blockIds: string[],
  attr: StrategyAttr,
  step: number,
): Record<string, BlockState> {
  const state: Record<string, BlockState> = Object.fromEntries(
    blockIds.map((b) => [b, { hits: [], restoredAt: null } as BlockState]),
  );
  for (let s = 0; s <= step && s < telemetry.length; s++) {
    // Order inside a step matches the harness: inject, then repair
    for (const h of faults[s] ?? []) state[h.b]?.hits.push({ h, step: s, sev: severity(h) });
    const t = telemetry[s];
    let restored: string[] = [];
    if (attr === "acc_periodic_scrub" && t.scrub_executed) restored = blockIds;
    if (attr === "acc_aegis") restored = t.aegis_restored_blocks;
    for (const b of restored) if (state[b]) state[b] = { hits: [], restoredAt: s };
  }
  return state;
}
