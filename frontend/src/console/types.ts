// Shapes sent by the console engine (aegis/console/engine.py, mission.py, server.py)

export type Num = number | "NaN" | "Infinity" | "-Infinity";
export type Severity = "critical" | "moderate" | "minor";
export type BitField = "exponent_msb" | "exponent" | "mantissa" | "sign" | "any";

export interface Hit {
  b: string;
  p: string;
  bit: number;
  from: Num;
  to: Num;
  mbu: boolean;
  sev: Severity;
  index?: number;
  field?: BitField;
  from_bits?: string;
  to_bits?: string;
  /** Minute of the mission at which the hit struck (sampled from the step's flux profile). */
  t_min?: number;
}

export interface BlockInfo {
  id: string;
  params: number;
  bytes: number;
  crc_ok: boolean;
  faults: number;
  worst: Severity | null;
  recent: Hit[];
  restored_at: number | null;
}

export interface Snapshot {
  status: "idle" | "booting" | "ready" | "error";
  boot_stage: string;
  boot_error: string | null;
  archs: string[];
  threads: number;
  mission_running: boolean;
  arch?: string;
  arch_label?: string;
  armed?: boolean;
  blocks?: BlockInfo[];
  clean_acc?: number;
  live_acc?: number;
  watch?: WatchChip[];
  probe_size?: number;
  gallery_size?: number;
  strikes?: number;
  repairs?: number;
  bytes_restored?: number;
  model_bytes?: number;
  golden_bytes?: number;
  corrupted?: string[];
}

export interface WatchChip {
  index: number;
  label: number;
  pred: number;
  conf: Num;
  golden_pred: number;
}

export interface Measure {
  stage: "after_strike" | "after_repair";
  accuracy: number;
  clean: number;
  correct: number;
  total: number;
  watch: WatchChip[];
  ms: number;
}

export interface T1Violation {
  layer: string;
  block: string;
  nan: boolean;
  inf: boolean;
  max: Num;
  min: Num;
  lo: Num;
  hi: Num;
}

export interface Diagnosis {
  healthy: boolean;
  suspects: string[];
  confirmed: string[];
  primary: string | null;
  t1: T1Violation[];
  t1_count: number;
  t1_layers: number;
  t2: { layer: string; block: string; score: Num; threshold: Num }[];
  t2_count: number;
  t2_layers: number;
  false_alarm: boolean;
  silent: string[];
  ms: number;
}

export interface Repair {
  mode: "aegis" | "audit" | "scrub";
  blocks: string[];
  bytes: number;
  energy_uj: Num;
  ms: Num;
  full_bytes: number;
  verified: boolean;
  still_corrupted: string[];
}

export interface Gallery {
  classes: string[];
  chips: { index: number; label: number; probe: boolean }[];
  watch: number[];
}

export interface TopK {
  cls: number;
  p: Num;
}

export interface ClassifyResult {
  results: {
    index: number;
    label: number;
    path: string;
    live: { pred: number | null; top: TopK[]; finite: boolean };
    golden: { pred: number; top: TopK[] };
  }[];
  ms: number;
  corrupted: string[];
}

export interface Evaluation {
  scope: "probe" | "full";
  n: number;
  live: number;
  golden: number;
  per_class: { class: string; n: number; live: number | null; golden: number | null }[];
  confusion: number[][];
  ms: number;
}

export type StrategyKey = "unmitigated" | "scrub" | "aegis" | "audit";

export interface MissionConfig {
  hours: number;
  step_minutes: number;
  flux: number;
  scrub_hours: number;
  audit_hours: number;
  probe_per_class: number;
  seed: number;
}

export interface Track {
  minutes: number[];
  lat: number[];
  lon: number[];
  alt_km: number;
  period_min: number;
}

export interface Field {
  lons: number[];
  lats: number[];
  b_nt: number[][];
  edge_nt: number;
  core_nt: number;
}

export interface StrategyStep {
  acc: number;
  mb: number;
  corrupted: string[];
  alarm: boolean;
  suspects: string[];
  restored: string[];
  scrubbed: boolean;
  audited: boolean;
}

export interface MissionStep {
  step: number;
  total_steps: number;
  t_min: number;
  in_saa: boolean;
  saa_minutes: number;
  flux_x: number;
  flips: number;
  msb: number;
  hits: Hit[];
  /** Strike minute of every flip in the step (live missions only). */
  hit_minutes?: number[];
  strategies: Record<StrategyKey, StrategyStep>;
  wall_s: number;
}

export interface MissionSummary {
  stopped: boolean;
  steps: number;
  hours: number;
  wall_s: number;
  flips: number;
  msb: number;
  audits: number;
  scrubs: number;
  strategies: Record<
    StrategyKey,
    {
      label: string;
      mean_acc: number | null;
      final_acc: number | null;
      min_acc: number | null;
      mb: number;
      mb_per_day: number;
      repairs: number;
      alarms: number;
    }
  >;
}

export interface MissionMeta {
  id: number;
  config: MissionConfig & { total_steps: number };
  strategies: { key: StrategyKey; label: string }[];
  blocks: { id: string; bytes: number }[];
  arch: string;
  arch_label: string;
  clean_acc: number;
  probe_size: number;
  expected_flips_per_day: number;
  full_model_mb: number;
  track: Track;
  field: Field | null;
  started: number;
  summary?: MissionSummary;
  error?: string;
}

export interface MissionState extends MissionMeta {
  steps: MissionStep[];
  running: boolean;
  stopping?: boolean;
}

export interface LogLine {
  t: number;
  level: "info" | "ok" | "warn" | "crit" | "error";
  tag: string;
  text: string;
}

export interface EngineEvent<T = unknown> {
  type: string;
  seq: number;
  t: number;
  data: T;
}
