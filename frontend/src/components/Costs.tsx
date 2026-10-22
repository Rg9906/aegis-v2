import { motion } from "motion/react";
import type { Campaign, StatRow } from "../lib/types";
import { mb, pct } from "../lib/format";
import { rowColor } from "../lib/data";
import SectionHead from "./SectionHead";

const KEYS = ["aegis_v2", "ranger_aegis", "aegis_audit", "periodic_scrubbing", "scrub_6h", "scrub_3h", "scrub_1h", "ground_uplink"];

function Bars({ rows, value, unit, digits }: { rows: StatRow[]; value: (r: StatRow) => number | null; unit: string; digits: (v: number) => string }) {
  const max = Math.max(...rows.map((r) => value(r) ?? 0), 1e-9);
  return (
    <ul className="hbars">
      {rows.map((r, i) => {
        const v = value(r);
        return (
          <li key={r.key}>
            <span className="hbar-label">{r.label}</span>
            <span className="hbar-track">
              {v !== null && (
                <motion.span
                  className="hbar-fill"
                  style={{ background: rowColor(r.key) }}
                  initial={{ width: 0 }}
                  whileInView={{ width: `${Math.max((100 * v) / max, 0.6)}%` }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.9, delay: i * 0.05, ease: [0.2, 0.7, 0.1, 1] }}
                />
              )}
            </span>
            <span className="hbar-val mono tabular">{v === null ? "—" : `${digits(v)} ${unit}`}</span>
          </li>
        );
      })}
    </ul>
  );
}

export default function Costs({ campaign }: { campaign: Campaign }) {
  const rows = KEYS.map((k) => campaign.stats.rows.find((r) => r.key === k)).filter((r): r is StatRow => !!r);
  const aegis = rows.find((r) => r.key === "aegis_v2");
  const scrub = rows.find((r) => r.key === "periodic_scrubbing");
  const verdict = campaign.stats.verdicts.find((v) => v.label.startsWith("Aegis-v2 selective"));
  const energyX = aegis?.energy_mj_day && scrub?.energy_mj_day ? aegis.energy_mj_day / scrub.energy_mj_day : null;

  return (
    <section className="section" id="costs">
      <SectionHead
        index="05"
        kicker="What it costs"
        title={<>Bandwidth is the win. <em>Energy is the price.</em></>}
        lede="Every strategy that repairs memory pays twice: bytes read from the golden copy, and energy for the reads plus any always-on monitoring. Aegis-v2 moves far fewer bytes than scrubbing at equal accuracy, but its always-on monitoring costs more energy than a slow scrub timer. Both sides are shown."
      />
      <div className="costs">
        <div className="panel costs-panel">
          <div className="costs-head">
            <h3>Repair traffic</h3>
            <span className="mono dim">MB per day from the golden copy</span>
          </div>
          <Bars rows={rows} value={(r) => r.mb_per_day} unit="MB" digits={mb} />
        </div>
        <div className="panel costs-panel">
          <div className="costs-head">
            <h3>Energy</h3>
            <span className="mono dim">mJ per day, memory reads + monitor compute</span>
          </div>
          <Bars rows={rows} value={(r) => r.energy_mj_day} unit="mJ" digits={(v) => v.toFixed(v >= 100 ? 0 : 1)} />
        </div>
      </div>
      <div className="costs-verdict">
        <div className="verdict-card verdict-good">
          <span className="mono">bandwidth</span>
          <b>{verdict?.traffic_ratio ? `${verdict.traffic_ratio.toFixed(1)}× less` : "—"}</b>
          <span>traffic than the scrub interval that would match Aegis-v2's {aegis ? pct(aegis.mean) : ""} mean accuracy</span>
        </div>
        <div className="verdict-card verdict-bad">
          <span className="mono">energy</span>
          <b>{energyX ? `${energyX.toFixed(1)}× more` : "—"}</b>
          <span>than 12-hour scrubbing; most of it is always-on monitoring (T1/T2 hook compute at an assumed 6.9 pJ/op, plus T3 checksum reads), not repair</span>
        </div>
        <div className="verdict-card">
          <span className="mono">accuracy</span>
          <b>{aegis && scrub ? `+${(100 * (aegis.mean - scrub.mean)).toFixed(1)} pts` : "—"}</b>
          <span>mean accuracy over 12-hour scrubbing, which spends most of each interval running on a corrupted model</span>
        </div>
      </div>
    </section>
  );
}
