import { useEffect, useId, useState } from "react";
import { motion } from "motion/react";
import type { Campaign } from "../lib/types";
import { accuracySeries, archLabel, loadCampaign, seedKeys } from "../lib/data";
import { pct } from "../lib/format";
import SectionHead from "./SectionHead";

const PAIR = ["main_final", "main_final_mobilenet"];
const ROWS = [
  ["aegis_v2", "Aegis-v2"],
  ["ranger_aegis", "Ranger + Aegis-v2"],
  ["aegis_audit", "Aegis-v2 + audit"],
  ["ranger_clipping", "Ranger clipping"],
  ["ground_uplink", "Ground uplink"],
  ["periodic_scrubbing", "Scrubbing 12 h"],
  ["blink_recentering", "BLINK"],
  ["unmitigated", "Unmitigated"],
] as const;
const COLORS = ["var(--accent)", "var(--s-acc_ranger_aegis)"];
const cache = new Map<string, Promise<Campaign>>();
const load = (n: string) => {
  if (!cache.has(n)) cache.set(n, loadCampaign(n));
  return cache.get(n)!;
};

function Spark({ c, color }: { c: Campaign; color: string }) {
  const clip = `spark-${useId().replace(/:/g, "")}`;
  const W = 300;
  const H = 70;
  const a = accuracySeries(c, "acc_aegis", "mean");
  const u = accuracySeries(c, "acc_unmitigated", "mean");
  const x = (i: number) => (i / (a.length - 1)) * W;
  const y = (v: number) => H - 4 - v * (H - 8);
  const line = (s: { mean: number }[]) => s.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p.mean).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="spark" aria-hidden="true">
      <defs>
        <clipPath id={clip}>
          <motion.rect x="0" y="0" height={H} initial={{ width: 0 }} whileInView={{ width: W }} viewport={{ once: true }} transition={{ duration: 1.6, ease: [0.2, 0.7, 0.1, 1] }} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        <path d={`${line(u)}L${W} ${H}L0 ${H}Z`} className="spark-area-bad" />
        <path d={line(u)} className="spark-bad" />
        <path d={line(a)} stroke={color} className="spark-good" />
      </g>
    </svg>
  );
}

export default function Architectures({ campaigns }: { campaigns: string[] }) {
  const [pair, setPair] = useState<Campaign[] | null>(null);
  const available = PAIR.every((n) => campaigns.includes(n));

  useEffect(() => {
    if (!available) return;
    Promise.all(PAIR.map(load)).then(setPair).catch(() => setPair(null));
  }, [available]);

  if (!available || !pair) return null;

  const mean = (c: Campaign, k: string) => c.stats.rows.find((r) => r.key === k)?.mean ?? null;
  const facts = (c: Campaign) => {
    const seeds = seedKeys(c).map((k) => c.seeds[k]);
    const flips = seeds.reduce((a, s) => a + s.summary.total_faults_accumulated, 0) / seeds.length / c.setup.days;
    const v = c.stats.verdicts.find((x) => x.label.startsWith("Aegis-v2 selective"));
    return [
      ["weights", `${(c.blocks.reduce((a, b) => a + b.bytes, 0) / 1e6).toFixed(1)} MB`],
      ["blocks", String(c.blocks.length)],
      ["clean accuracy", pct(c.setup.clean_accuracy)],
      ["bit flips / day", flips.toFixed(0)],
      ["Aegis-v2 traffic", `${v?.mb_per_day.toFixed(1)} MB/day`],
      ["scrub traffic to match", v?.traffic_ratio ? `${v.traffic_ratio.toFixed(1)}×` : "—"],
    ];
  };

  return (
    <section className="section" id="models">
      <SectionHead
        index="07"
        kicker="Two architectures"
        title={<>A different brain, <em>the same winner.</em></>}
        lede={`The same orbit, radiation model and seeds, applied to a ${facts(pair[0])[0][1]} ResNet-18 and a ${facts(pair[1])[0][1]} MobileNetV2 fine-tuned on chips held out of every campaign split. Fewer stored bits means fewer hits. Aegis-v2 stays on top for both; the baselines reshuffle, with Ranger clipping, weak on ResNet-18, nearly matching it on MobileNetV2.`}
      />
      <div className="arch-grid">
        {pair.map((c, i) => (
          <motion.div
            key={c.name}
            className="panel arch-card"
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6, delay: i * 0.1 }}
          >
            <div className="arch-top">
              <span className="swatch" style={{ background: COLORS[i] }} />
              <h3>{archLabel(c.setup.arch)}</h3>
            </div>
            <dl className="arch-facts">
              {facts(c).map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd className="mono tabular">{v}</dd>
                </div>
              ))}
            </dl>
            <Spark c={c} color={COLORS[i]} />
            <div className="spark-legend mono dim">
              <span><i style={{ background: COLORS[i] }} /> Aegis-v2</span>
              <span><i className="spark-legend-bad" /> unmitigated</span>
              <span>mean of {c.stats.seeds} seeds · 3 days</span>
            </div>
          </motion.div>
        ))}

        <div className="panel arch-dumbbell">
          <div className="costs-head">
            <h3>Mean accuracy under radiation</h3>
            <span className="mono dim">
              <span className="swatch" style={{ background: COLORS[0] }} /> ResNet-18 &nbsp;
              <span className="swatch" style={{ background: COLORS[1] }} /> MobileNetV2
            </span>
          </div>
          <div className="db-axis mono dim">
            {[0, 25, 50, 75, 100].map((t) => (
              <span key={t} style={{ left: `${t}%` }}>{t}%</span>
            ))}
          </div>
          {ROWS.map(([k, label], i) => {
            const a = mean(pair[0], k);
            const b = mean(pair[1], k);
            if (a === null || b === null) return null;
            return (
              <div key={k} className="db-row">
                <span className="db-label">{label}</span>
                <span className="db-track">
                  <motion.span
                    className="db-link"
                    initial={{ left: "0%", width: 0 }}
                    whileInView={{ left: `${100 * Math.min(a, b)}%`, width: `${100 * Math.abs(a - b)}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.9, delay: i * 0.05 }}
                  />
                  {[a, b].map((v, j) => (
                    <motion.span
                      key={j}
                      className="db-dot"
                      style={{ background: COLORS[j] }}
                      initial={{ left: "0%" }}
                      whileInView={{ left: `${100 * v}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.9, delay: i * 0.05, ease: [0.2, 0.7, 0.1, 1] }}
                      title={`${archLabel(pair[j].setup.arch)}: ${pct(v)}`}
                    />
                  ))}
                </span>
                <span className="db-vals mono tabular">
                  {pct(a, 0)} · {pct(b, 0)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
