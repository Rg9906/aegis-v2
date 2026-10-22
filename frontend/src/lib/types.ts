// Shapes written by experiments/export_frontend_data.py

export type StrategyAttr =
  | "acc_aegis"
  | "acc_periodic_scrub"
  | "acc_ranger"
  | "acc_blink"
  | "acc_unmitigated"
  | "acc_uplink"
  | "acc_ranger_aegis"
  | "acc_aegis_audit";

export interface Strategy {
  attr: StrategyAttr;
  label: string;
  light: string;
  dark: string;
}

export interface FaultHit {
  b: string; // block id
  p: string; // parameter name
  bit: number;
  from: number;
  to: number | string; // string for inf / nan
  mbu: boolean;
}

export interface TelemetryStep {
  step_index: number;
  elapsed_days: number;
  in_saa: boolean;
  saa_minutes_in_step: number;
  bit_flips_step: number;
  exponent_msb_flips_step: number;
  aegis_alarm: boolean;
  aegis_restored_blocks: string[];
  scrub_executed: boolean;
  acc: Partial<Record<StrategyAttr, number>>;
  acc_scrub_sweep: Record<string, number>;
}

export interface SeedRun {
  summary: {
    mean_accuracy: Record<string, number>;
    final_accuracy: Record<string, number>;
    bytes_transferred: Record<string, number>;
    total_faults_accumulated: number;
    exponent_msb_flips_total: number;
  };
  telemetry: TelemetryStep[];
  faults: FaultHit[][];
}

export interface StatRow {
  key: string;
  label: string;
  mean: number;
  ci: number;
  final: number | null;
  final_ci: number | null;
  mb_per_day: number;
  energy_mj_day: number | null;
  energy_ci: number | null;
  per_seed: number[];
  delta_vs_aegis: { mean: number; ci: number; wins: number } | null;
  interval_h?: number;
}

export interface ParetoPoint {
  strategy: string;
  mean_accuracy: number;
  mb_per_day: number;
  family: "aegis" | "scrub" | "uplink" | "baseline";
  interval_h?: number;
}

export interface Verdict {
  label: string;
  mean_accuracy: number;
  mb_per_day: number;
  dominated_by: string[];
  scrub_mb_to_match: number | null;
  traffic_ratio: number | null;
}

export interface Campaign {
  name: string;
  generated: string;
  setup: {
    arch: string;
    orbit: string;
    days: number;
    steps: number;
    step_minutes: number;
    fault_field: string;
    clean_accuracy: number;
    golden_int8: boolean;
    audit_hours: number | null;
    rates: { background: number; saa_core: number; measured_mean: number; sbu_fraction: number };
    energy: { monitor_pj_per_op: number; dram_read_pj_per_bit: number; inference_mj_per_image: number };
  };
  strategies: Strategy[];
  blocks: { id: string; bytes: number }[];
  stats: {
    seeds: number;
    rows: StatRow[];
    pareto: ParetoPoint[];
    verdicts: Verdict[];
    energy_hook_compute_added: boolean;
  };
  seeds: Record<string, SeedRun>;
  track: { minutes: number[]; lat: number[]; lon: number[]; alt_km: number; period_min: number } | null;
  field: { lons: number[]; lats: number[]; b_nt: number[][]; edge_nt: number; core_nt: number } | null;
}
