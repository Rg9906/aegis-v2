import { motion } from "motion/react";
import { blockShort } from "../../lib/format";
import { fmtValue, mbStr } from "../fx";
import type { Diagnosis, Repair } from "../types";

export type FunnelPhase = "idle" | "scanning" | "diagnosed" | "repairing" | "repaired";

interface Props {
  phase: FunnelPhase;
  diagnosis: Diagnosis | null;
  repair: Repair | null;
  armed: boolean;
}

type StageState = "idle" | "scan" | "alarm" | "clear" | "done" | "skip";

function Stage({ n, title, state, delay, children }: { n: string; title: string; state: StageState; delay: number; children: React.ReactNode }) {
  return (
    <motion.div
      className={`fn-stage st-${state}`}
      initial={false}
      animate={{ opacity: state === "idle" ? 0.55 : 1 }}
      transition={{ delay: state === "idle" ? 0 : delay, duration: 0.35 }}
    >
      <div className="fn-head">
        <span className="fn-n mono">{n}</span>
        <span className="fn-title">{title}</span>
        <span className="fn-light" aria-hidden="true" />
      </div>
      <motion.div
        className="fn-body"
        key={state}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: state === "idle" || state === "scan" ? 0 : delay, duration: 0.35 }}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

const Chips = ({ ids, tone }: { ids: string[]; tone: string }) => (
  <span className="fn-chips">
    {ids.map((b) => (
      <span key={b} className={`fn-chip ${tone} mono`}>
        {blockShort(b)}
      </span>
    ))}
  </span>
);

/** T1 -> T2 -> T3 -> restore, lighting up stage by stage as the real report arrives. */
export default function DetectionFunnel({ phase, diagnosis, repair, armed }: Props) {
  const d = phase === "idle" || phase === "scanning" ? null : diagnosis;
  const scanning = phase === "scanning";
  const t1: StageState = scanning ? "scan" : !d ? "idle" : d.t1_count ? "alarm" : "clear";
  const t2: StageState = scanning ? "scan" : !d ? "idle" : d.t2_count ? "alarm" : "clear";
  const t3: StageState = scanning ? "scan" : !d ? "idle" : d.confirmed.length ? "alarm" : d.suspects.length || d.silent.length ? "done" : "clear";
  const rs: StageState =
    phase === "repairing" ? "scan" : phase === "repaired" && repair ? "done" : d && !d.healthy ? (armed ? "scan" : "skip") : "idle";
  const topT1 = d?.t1[0];
  const topT2 = d?.t2.slice().sort((a, b) => Number(b.score) - Number(a.score))[0];

  return (
    <div className="funnel">
      <Stage n="T1" title="Activation envelope" state={t1} delay={0}>
        {t1 === "scan" && <p className="dim">Hooks on every conv output…</p>}
        {t1 === "idle" && <p className="dim">Clean min/max bounds from 100 chips</p>}
        {d && (
          <>
            <p className="fn-big">
              <strong className="mono">{d.t1_count}</strong>
              <span className="dim"> / {d.t1_layers} layers out of bounds</span>
            </p>
            {topT1 && (
              <p className="fn-detail mono">
                {topT1.layer}: {topT1.nan ? "NaN" : topT1.inf ? "∞" : fmtValue(topT1.max)} <span className="dim">vs ≤ {fmtValue(topT1.hi)}</span>
              </p>
            )}
          </>
        )}
      </Stage>
      <Stage n="T2" title="BatchNorm drift" state={t2} delay={0.25}>
        {t2 === "scan" && <p className="dim">Batch statistics vs running stats…</p>}
        {t2 === "idle" && <p className="dim">Per-layer tolerance from clean batches</p>}
        {d && (
          <>
            <p className="fn-big">
              <strong className="mono">{d.t2_count}</strong>
              <span className="dim"> / {d.t2_layers} BN layers drifting</span>
            </p>
            {topT2 && (
              <p className="fn-detail mono">
                {topT2.layer}: {fmtValue(topT2.score)} <span className="dim">vs {fmtValue(topT2.threshold)}</span>
              </p>
            )}
          </>
        )}
      </Stage>
      <Stage n="T3" title="CRC-32 confirm" state={t3} delay={0.5}>
        {t3 === "scan" && <p className="dim">Checksumming suspect blocks…</p>}
        {t3 === "idle" && <p className="dim">Bit-exact check of suspects only</p>}
        {d && d.suspects.length > 0 && (
          <div className="fn-flow">
            <span className="dim">suspects</span>
            <Chips ids={d.suspects} tone="amber" />
            <span className="dim">confirmed</span>
            {d.confirmed.length ? <Chips ids={d.confirmed} tone="red" /> : <span className="fn-ok">none · false alarm absorbed</span>}
          </div>
        )}
        {d && !d.suspects.length && !d.silent.length && <p className="fn-ok">No suspects: payload healthy</p>}
        {d && d.silent.length > 0 && (
          <div className="fn-flow">
            <span className="fn-warn">Silent corruption below behavioral threshold in</span>
            <Chips ids={d.silent} tone="amber" />
            <span className="dim">only a full CRC audit catches this</span>
          </div>
        )}
      </Stage>
      <Stage n="R" title="Selective restore" state={rs} delay={0.75}>
        {rs === "idle" && <p className="dim">Reload only confirmed blocks from the zstd golden store</p>}
        {rs === "skip" && <p className="fn-warn">Autopilot off: damage persists until you repair</p>}
        {rs === "scan" && phase !== "repaired" && <p className="dim">Streaming golden bytes…</p>}
        {repair && phase === "repaired" && (
          <>
            <p className="fn-big">
              <strong className="mono">{mbStr(repair.bytes)} MB</strong>
              <span className="dim"> of {mbStr(repair.full_bytes)} MB</span>
            </p>
            <div className="fn-bar" aria-hidden="true">
              <motion.span initial={{ width: 0 }} animate={{ width: `${Math.max(1.5, (100 * repair.bytes) / repair.full_bytes)}%` }} transition={{ duration: 0.8 }} />
            </div>
            <p className="fn-detail mono">
              <Chips ids={repair.blocks.length > 6 ? repair.blocks.slice(0, 6) : repair.blocks} tone="blue" />
              {repair.blocks.length > 6 && <span className="dim"> +{repair.blocks.length - 6}</span>} · {fmtValue(repair.ms)} ms ·{" "}
              {repair.verified ? <span className="fn-ok">CRC verified</span> : <span className="fn-warn">CRC mismatch</span>}
            </p>
          </>
        )}
      </Stage>
    </div>
  );
}
