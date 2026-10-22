import { useMemo, useState } from "react";
import { motion } from "motion/react";
import type { Campaign, FaultHit } from "../lib/types";
import { seedKeys, severity, type Severity } from "../lib/data";
import { blockShort, missionClock } from "../lib/format";
import SectionHead from "./SectionHead";

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const bitsOf = (x: number) => ((f32[0] = x), u32[0]);
const valueOf = (bits: number) => ((u32[0] = bits >>> 0), f32[0]);

function fmt(x: number) {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x > 0 ? "+∞" : "−∞";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a >= 1e4 || a < 1e-3 ? x.toExponential(3) : x.toPrecision(5);
}

const field = (bit: number) => (bit === 31 ? "sign" : bit >= 23 ? "exponent" : "mantissa");

function severityOf(from: number, to: number, p: string, bit: number): Severity {
  return severity({ b: "", p, bit, from, to: Number.isFinite(to) ? to : "nan", mbu: false });
}

interface Pick {
  h: FaultHit;
  seed: string;
  step: number;
}

/** A spread of real hits from this campaign: the worst, a sign flip, a moderate and a harmless one. */
function examples(c: Campaign): { picks: Pick[]; counts: Record<string, number>; total: number } {
  const all: Pick[] = seedKeys(c).flatMap((seed) =>
    c.seeds[seed].faults.flatMap((fs, step) => fs.map((h) => ({ h, seed, step }))),
  );
  const counts: Record<string, number> = { sign: 0, exponent: 0, mantissa: 0 };
  for (const { h } of all) counts[field(h.bit)]++;
  const ratio = (p: Pick) =>
    typeof p.h.to === "string" ? Infinity : Math.abs(p.h.to) / Math.max(Math.abs(p.h.from), 1e-30);
  const byRatio = [...all].sort((a, b) => ratio(b) - ratio(a));
  const picks: Pick[] = [];
  const add = (p?: Pick) => p && !picks.includes(p) && picks.push(p);
  add(byRatio.find((p) => p.h.bit === 30 && typeof p.h.to === "number"));
  add(byRatio.find((p) => p.h.bit === 31 && p.h.p.endsWith("running_var")));
  add(byRatio.find((p) => p.h.bit === 31));
  add(all.find((p) => severity(p.h) === "moderate"));
  add(all.find((p) => p.h.bit < 23 && severity(p.h) === "minor"));
  add(byRatio[0]);
  return { picks: picks.slice(0, 5), counts, total: all.length };
}

const SEV_LABEL: Record<Severity, string> = { critical: "destructive", moderate: "moderate", minor: "negligible" };

export default function Anatomy({ campaign }: { campaign: Campaign }) {
  const { picks, counts, total } = useMemo(() => examples(campaign), [campaign]);
  const [sel, setSel] = useState(0);
  const pick = picks[sel];
  const base = pick ? bitsOf(pick.h.from) : bitsOf(0.106);
  const [bits, setBits] = useState<number>(() => (pick ? (base ^ (1 << pick.h.bit)) >>> 0 : base));

  const choose = (i: number) => {
    setSel(i);
    const p = picks[i];
    setBits((bitsOf(p.h.from) ^ (1 << p.h.bit)) >>> 0);
  };

  const from = valueOf(base);
  const to = valueOf(bits);
  const changed = (bits ^ base) >>> 0;
  // Grade the bits actually changed now, not the recorded hit (the user may have toggled others)
  const sev: Severity = changed === 0 ? "minor" : severityOf(from, to, pick?.h.p ?? "", (changed >>> 30) & 1 ? 30 : -1);
  const ratio = Number.isFinite(to) && from !== 0 ? Math.abs(to / from) : Infinity;

  // log10 magnitude gauge from 1e-8 to 1e39
  const gx = (x: number) => {
    if (!Number.isFinite(x)) return 100;
    const l = Math.log10(Math.max(Math.abs(x), 1e-8));
    return Math.max(0, Math.min(100, ((l + 8) / 47) * 100));
  };

  return (
    <section className="section" id="anatomy">
      <SectionHead
        index="02"
        kicker="Anatomy of an upset"
        title={<>One bit, <em>thirty-two</em> ways to fail.</>}
        lede="Every weight is a 32-bit IEEE-754 float. Most flips land in the mantissa and barely move the value; a flip in the top exponent bit multiplies it by 2¹²⁸. These are real hits from this campaign. Click any bit to flip it yourself."
      />
      <div className="anatomy">
        <div className="panel anatomy-main">
          <div className="chips" role="tablist" aria-label="Recorded faults">
            {picks.map((p, i) => (
              <button key={i} role="tab" aria-selected={i === sel} className={`chip ${i === sel ? "chip-on" : ""}`} onClick={() => choose(i)}>
                <span className={`sev-dot sev-${severity(p.h)}`} />
                {blockShort(p.h.b)} · bit {p.h.bit}
              </button>
            ))}
          </div>
          {pick && (
            <p className="anatomy-where mono dim">
              seed {pick.seed} · {missionClock((pick.step + 1) * campaign.setup.step_minutes)} · <span className="ink">{pick.h.p}</span>
            </p>
          )}

          <div className="bitrow" role="group" aria-label="32 stored bits, sign first">
            {Array.from({ length: 32 }, (_, k) => {
              const bit = 31 - k;
              const on = (bits >>> bit) & 1;
              const flipped = (changed >>> bit) & 1;
              return (
                <motion.button
                  key={bit}
                  className={`bit bit-${field(bit)} ${flipped ? "bit-flipped" : ""} ${pick?.h.bit === bit ? "bit-origin" : ""}`}
                  onClick={() => setBits((b) => (b ^ (1 << bit)) >>> 0)}
                  aria-label={`bit ${bit} (${field(bit)}) is ${on}${flipped ? ", flipped" : ""}`}
                  whileTap={{ scale: 0.85 }}
                  animate={flipped ? { y: [0, -6, 0] } : { y: 0 }}
                  transition={{ duration: 0.35 }}
                >
                  {on}
                </motion.button>
              );
            })}
          </div>
          <div className="bitlegend mono">
            <span style={{ flex: 1 }} title="sign">±</span>
            <span style={{ flex: 8 }}>exponent · ×2^(e−127)</span>
            <span style={{ flex: 23 }}>mantissa · 1.fraction</span>
          </div>

          <div className="anatomy-values">
            <div>
              <div className="kpi-label">stored before</div>
              <div className="anatomy-num tabular">{fmt(from)}</div>
            </div>
            <div className="anatomy-arrow" aria-hidden="true">→</div>
            <div>
              <div className="kpi-label">after the upset</div>
              <motion.div key={bits} className={`anatomy-num tabular sev-text-${sev}`} initial={{ opacity: 0.2, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
                {fmt(to)}
              </motion.div>
            </div>
            <div className="anatomy-sev">
              <span className={`status status-${sev === "critical" ? "retracted" : sev === "moderate" ? "mitigated" : "fixed"}`}>{SEV_LABEL[sev]}</span>
              <div className="mono dim">{changed === 0 ? "no change" : Number.isFinite(ratio) ? `|×| ${ratio >= 1e3 || ratio < 1e-3 ? ratio.toExponential(1) : ratio.toFixed(3)}` : "non-finite"}</div>
            </div>
          </div>

          <div className="gauge" aria-hidden="true">
            <div className="gauge-track" />
            {[-8, 0, 8, 16, 24, 32].map((e) => (
              <span key={e} className="gauge-tick mono" style={{ left: `${((e + 8) / 47) * 100}%` }}>
                1e{e}
              </span>
            ))}
            <motion.span className="gauge-mark gauge-from" animate={{ left: `${gx(from)}%` }} />
            <motion.span className={`gauge-mark gauge-to sev-bg-${sev}`} animate={{ left: `${gx(to)}%` }} transition={{ type: "spring", stiffness: 120, damping: 16 }} />
          </div>
          <div className="gauge-cap mono dim">magnitude, log scale · ○ before · ● after</div>
        </div>

        <div className="panel anatomy-side">
          <h3>Where {total} upsets landed</h3>
          <p className="dim">Faults fall uniformly over stored bits, so the field sizes set the odds.</p>
          <div className="fieldbar">
            {(["sign", "exponent", "mantissa"] as const).map((f) => (
              <motion.div
                key={f}
                className={`fieldbar-seg bit-${f}`}
                initial={{ flexGrow: 0 }}
                whileInView={{ flexGrow: Math.max(counts[f], 0.4) }}
                viewport={{ once: true }}
                transition={{ duration: 1.1, ease: [0.2, 0.7, 0.1, 1] }}
                title={`${f}: ${counts[f]}`}
              />
            ))}
          </div>
          <ul className="fieldlist">
            {(["mantissa", "exponent", "sign"] as const).map((f) => (
              <li key={f}>
                <span className={`swatch swatch-sq bit-${f}`} />
                <span>{f}</span>
                <span className="mono tabular">{counts[f]}</span>
                <span className="mono dim tabular">{total ? ((100 * counts[f]) / total).toFixed(0) : 0}%</span>
              </li>
            ))}
          </ul>
          <div className="anatomy-note">
            <b>Why the gate matters.</b> Most of these hits change nothing a classifier can feel. Reloading the whole model
            for them is wasted traffic; ignoring the rare exponent-MSB hit is fatal. Aegis-v2 acts only when behavior and
            checksums both say a block is broken.
          </div>
        </div>
      </div>
    </section>
  );
}
