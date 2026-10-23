import { AnimatePresence, motion } from "motion/react";
import { blockShort } from "../../lib/format";
import { STRATEGY_COLOR } from "../constants";
import { AnimatedNumber } from "../fx";
import type { MissionState, StrategyKey } from "../types";

const BLURB: Record<StrategyKey, string> = {
  unmitigated: "No protection: every flip stays",
  scrub: "Blind full reload on a timer",
  aegis: "Detect, CRC-confirm, restore only broken blocks",
  audit: "Aegis-v2 plus a periodic full CRC scan",
};

/** One strategy's live lane: accuracy, sparkline, traffic, memory cells and latest action. */
export default function StrategyCard({ k, label, mission, maxMb }: { k: StrategyKey; label: string; mission: MissionState; maxMb: number }) {
  const steps = mission.steps;
  const last = steps[steps.length - 1]?.strategies[k];
  const acc = last?.acc ?? mission.clean_acc;
  const mb = last?.mb ?? 0;
  const hist = steps.map((s) => s.strategies[k].acc);
  const mean = hist.length ? hist.reduce((a, b) => a + b, 0) / hist.length : mission.clean_acc;
  const corrupted = new Set(last?.corrupted ?? []);
  const action = !last
    ? null
    : last.scrubbed
      ? { tone: "scrub", text: `Scrub · full reload` }
      : last.restored.length
        ? { tone: "fix", text: `${last.audited && !last.alarm ? "Audit" : "Alarm"} → restored ${last.restored.map(blockShort).join(" ")}` }
        : last.audited
          ? { tone: "ok", text: "Audit · all CRCs match" }
          : last.alarm
            ? { tone: "warn", text: "Alarm · no CRC mismatch" }
            : corrupted.size
              ? { tone: "bad", text: `${corrupted.size} block${corrupted.size > 1 ? "s" : ""} corrupt` }
              : { tone: "ok", text: "Nominal" };
  const featured = k === "aegis" || k === "audit";

  // Sparkline
  const W = 120;
  const H = 30;
  const n = mission.config.total_steps;
  const pts = hist.map((v, i) => `${(i / Math.max(1, n - 1)) * W},${H - v * H}`).join(" ");

  return (
    <motion.article className={`strat ${featured ? "featured" : ""}`} style={{ ["--c" as string]: STRATEGY_COLOR[k] }} layout>
      <header className="strat-head">
        <span className="strat-swatch" aria-hidden="true" />
        <div>
          <h4>{label}</h4>
          <p className="dim">{BLURB[k]}</p>
        </div>
      </header>
      <div className="strat-main">
        <div className={`strat-acc ${acc < mission.clean_acc - 0.05 ? "down" : ""}`}>
          <AnimatedNumber value={acc * 100} suffix="%" duration={0.7} />
          <span className="strat-sub mono">mean {(mean * 100).toFixed(1)}%</span>
        </div>
        <svg className="strat-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <line x1="0" x2={W} y1={H - mission.clean_acc * H} y2={H - mission.clean_acc * H} className="spark-clean" />
          {hist.length > 1 && <polyline points={pts} className="spark-line" />}
        </svg>
      </div>
      <div className="strat-traffic">
        <span className="dim">repair traffic</span>
        <span className="mono">
          <AnimatedNumber value={mb} decimals={1} suffix=" MB" duration={0.6} />
        </span>
        <div className="strat-bar" aria-hidden="true">
          <motion.span animate={{ width: `${maxMb > 0 ? Math.max(mb > 0 ? 2 : 0, (100 * mb) / maxMb) : 0}%` }} transition={{ duration: 0.6 }} />
        </div>
      </div>
      <div className="strat-cells" aria-label={`${corrupted.size} corrupted blocks`}>
        {mission.blocks.map((b) => (
          <i key={b.id} className={corrupted.has(b.id) ? "bad" : last?.restored.includes(b.id) ? "fixed" : ""} title={b.id} />
        ))}
      </div>
      <div className="strat-action">
        <AnimatePresence mode="wait">
          {action && (
            <motion.span
              key={`${steps.length}-${action.text}`}
              className={`act act-${action.tone}`}
              initial={{ opacity: 0, y: 8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25 }}
            >
              {action.text}
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </motion.article>
  );
}
