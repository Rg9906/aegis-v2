import { useEffect, useRef, useState } from "react";
import { animate, motion, useInView } from "motion/react";
import type { Campaign } from "../lib/types";
import { isPreFix, seedKeys } from "../lib/data";
import Globe from "./Globe";

function CountUp({ to, decimals = 1, suffix = "" }: { to: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const c = animate(0, to, { duration: 1.4, ease: [0.2, 0.7, 0.1, 1], onUpdate: setV });
    return () => c.stop();
  }, [inView, to]);
  return (
    <span ref={ref} className="tabular">
      {v.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export default function Hero({ campaign, theme }: { campaign: Campaign; theme: string }) {
  const row = (k: string) => campaign.stats.rows.find((r) => r.key === k);
  const aegis = row("aegis_v2");
  const scrub = row("periodic_scrubbing");
  const verdict = campaign.stats.verdicts.find((v) => v.label.startsWith("Aegis-v2 selective"));
  const seeds = seedKeys(campaign).map((k) => campaign.seeds[k]);
  const flipsPerDay =
    seeds.reduce((a, s) => a + s.summary.total_faults_accumulated, 0) / seeds.length / campaign.setup.days;
  const { setup } = campaign;

  return (
    <section className="hero" id="overview">
      <div className="hero-copy">
        <motion.p
          className="overline"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Mission dossier · Sentinel-2A orbit · {setup.days} days · {campaign.stats.seeds} radiation seeds ·{" "}
          {setup.arch === "mobilenet_v2" ? "MobileNetV2" : "ResNet-18"}
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.05, ease: [0.2, 0.7, 0.1, 1] }}
        >
          A satellite brain
          <br />
          that <em>heals itself.</em>
        </motion.h1>
        <motion.p
          className="lede"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Radiation in low Earth orbit flips bits in an onboard AI model's memory. Aegis-v2 notices the damage
          from the model's own activity, confirms it bit-for-bit with checksums, and reloads only the broken
          layers, without waiting for a ground station.
        </motion.p>

        <div className="kpis">
          {aegis && (
            <div className="kpi kpi-hero">
              <div className="kpi-value">
                <CountUp to={aegis.mean * 100} suffix="%" />
              </div>
              <div className="kpi-label">
                mean accuracy under radiation
                <span className="kpi-sub">clean model: {(setup.clean_accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}
          {scrub && (
            <div className="kpi">
              <div className="kpi-value muted">
                <CountUp to={scrub.mean * 100} suffix="%" />
              </div>
              <div className="kpi-label">periodic scrubbing every 12 h</div>
            </div>
          )}
          {verdict?.traffic_ratio && (
            <div className="kpi">
              <div className="kpi-value">
                <CountUp to={verdict.traffic_ratio} suffix="×" />
              </div>
              <div className="kpi-label">more repair traffic scrubbing needs to match it</div>
            </div>
          )}
          <div className="kpi">
            <div className="kpi-value">
              <CountUp to={flipsPerDay} decimals={0} />
            </div>
            <div className="kpi-label">bit flips per day, calibrated to Alsat-1 flight data</div>
          </div>
        </div>

        {isPreFix(campaign) && (
          <p className="prefix-note">
            <span className="status status-mitigated">Pre-fix run</span> These numbers predate two fixes found in this
            campaign (see <a href="#incidents">incident files</a>); a corrected run is pending.
          </p>
        )}

        <div className="hero-actions">
          <a className="btn btn-primary" href="#replay">
            Replay the mission
          </a>
          <a className="btn" href="#tradeoff">
            See the trade-off
          </a>
        </div>
      </div>
      <motion.div
        className="hero-globe"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.4, ease: [0.2, 0.7, 0.1, 1] }}
      >
        <Globe campaign={campaign} theme={theme} />
        <div className="globe-legend">
          <span className="dot dot-accent" /> Sentinel-2A track
          <span className="dot dot-saa" /> South Atlantic Anomaly (IGRF |B|)
        </div>
      </motion.div>
    </section>
  );
}
