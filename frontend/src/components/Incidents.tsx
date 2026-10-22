import { motion } from "motion/react";
import SectionHead from "./SectionHead";

const CHAIN = ["stem", "L1·0", "L1·1", "L2·0", "L2·1", "L3·0", "L3·1", "L4·0", "L4·1", "FC"];

interface ChainProps {
  culprit?: string;
  alarms?: string[];
  restored?: string[];
}

/** Forward-order block chain: culprit (red), alarm sites (amber ring), restored (green). */
function Chain({ culprit, alarms = [], restored = [] }: ChainProps) {
  const cw = 46;
  const gap = 8;
  const W = CHAIN.length * (cw + gap) - gap;
  return (
    <svg viewBox={`-5 0 ${W + 10} 64`} className="chain" role="img" aria-label="Block chain diagram">
      <line x1={0} x2={W} y1={22} y2={22} className="chain-wire" />
      {CHAIN.map((b, i) => {
        const x = i * (cw + gap);
        const isCulprit = b === culprit;
        const isAlarm = alarms.includes(b);
        const isRestored = restored.includes(b);
        return (
          <g key={b}>
            <rect
              x={x}
              y={8}
              width={cw}
              height={28}
              rx={6}
              className={`chain-cell ${isCulprit ? "chain-culprit" : ""} ${isRestored ? "chain-restored" : ""}`}
            />
            {isAlarm && <rect x={x - 3} y={5} width={cw + 6} height={34} rx={8} className="chain-alarm" />}
            <text x={x + cw / 2} y={26} textAnchor="middle" className="chain-text">
              {b}
            </text>
            {isCulprit && (
              <text x={x + cw / 2} y={56} textAnchor="middle" className="chain-note crit-text">
                fault
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

interface Incident {
  id: string;
  status: "fixed" | "mitigated" | "retracted";
  title: string;
  when: string;
  what: string;
  why: string;
  fix: string;
  evidence: string;
  chain?: ChainProps;
}

const INCIDENTS: Incident[] = [
  {
    id: "INC-00",
    status: "retracted",
    title: "The headline that wasn't",
    when: "Prototype, before 2026-10-21",
    what: "An early build reported 98.9% energy savings over scrubbing.",
    why: "It evaluated on random-noise images with random labels, fed the model 64 px chips with alphabetical labels (5.6% accuracy, chance), and its activation envelope fired on 19 to 20 of 21 layers on every clean batch, so the detection gate restored everything.",
    fix: "Real EuroSAT chips at 224 px with the checkpoint's label order (92.8% clean), envelopes from observed clean min/max (0 false alarms), disjoint calibration and evaluation splits, recalibration energy charged.",
    evidence: "tests/test_correctness_fixes.py · every result on this page postdates the fix",
  },
  {
    id: "INC-01",
    status: "fixed",
    title: "The culprit one block upstream",
    when: "Seed 1 · T+0d 17:00",
    what: "A sign-bit hit turned layer3.0.bn2.running_var from 110 to −110. Batch norm takes its square root, so the block emitted NaN and accuracy fell to 10% (chance).",
    why: "The monitor watches conv outputs and batch-norm inputs, so the NaN first appeared one block later. Checksums confirmed the downstream blocks (they carried older, harmless flips) and restored those, while the real culprit was never checked.",
    fix: "On every alarm, also checksum the block just upstream of the earliest alarm. Replaying seed 1 with the fix holds 89.3% through the event.",
    evidence: "tests/test_monitors.py::test_bn_running_var_sign_flip_restores_its_own_block fails without the fix",
    chain: { culprit: "L3·0", alarms: ["L3·1", "L4·0", "L4·1", "FC"], restored: ["L3·1", "L4·0", "L4·1"] },
  },
  {
    id: "INC-02",
    status: "mitigated",
    title: "The fault nobody can feel",
    when: "Seed 1 · T+2d 01:00",
    what: "Bit 25 of one layer1.0.conv1 weight flipped: 0.106 became 1.70. Accuracy dropped from 89.3% to 72.0% and stayed there for the rest of the mission.",
    why: "A 16× weight is not large enough to push any activation outside its calibrated envelope or shift batch-norm statistics, so no behavioral alarm ever fired and the block was never checksummed.",
    fix: "Aegis-v2 + CRC audit: every 6 hours, checksum all blocks and restore any mismatch. It catches silent faults within one audit period at the cost of restoring some harmless flips too; the trade-off is measured as its own strategy.",
    evidence: "tests/test_baselines_and_sim.py::test_crc_audit_repairs_silent_faults",
    chain: { culprit: "L1·0" },
  },
];

const STATUS_LABEL = { fixed: "Fixed", mitigated: "Mitigated", retracted: "Retracted" };

/** `preFix`: the loaded campaign was run before INC-01 / INC-02 were addressed. */
export default function Incidents({ preFix }: { preFix: boolean }) {
  return (
    <section className="section" id="incidents">
      <SectionHead
        index="08"
        kicker="Incident files"
        title={<>What broke, and what we <em>changed</em>.</>}
        lede={
          preFix
            ? "Failures found while building and testing Aegis-v2, each traced to a root cause with the exact fault that triggered it. The campaign shown on this page was run BEFORE the INC-01 and INC-02 changes, so its Aegis-v2 numbers include both failures."
            : "Failures found while building and testing Aegis-v2, each traced to a root cause with the exact fault that triggered it. The campaign shown on this page includes every change listed here."
        }
      />
      <div className="incidents">
        {INCIDENTS.map((inc, i) => (
          <motion.article
            key={inc.id}
            className="panel incident"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6, delay: i * 0.08, ease: [0.2, 0.7, 0.1, 1] }}
          >
            <header className="incident-head">
              <span className="mono incident-id">{inc.id}</span>
              <span className={`status status-${inc.status}`}>{STATUS_LABEL[inc.status]}</span>
              <span className="incident-when mono">{inc.when}</span>
            </header>
            <h3>{inc.title}</h3>
            {inc.chain && <Chain {...inc.chain} />}
            <dl>
              <dt>What happened</dt>
              <dd>{inc.what}</dd>
              <dt>Root cause</dt>
              <dd>{inc.why}</dd>
              <dt>Change</dt>
              <dd>{inc.fix}</dd>
            </dl>
            <p className="evidence mono">{inc.evidence}</p>
          </motion.article>
        ))}
      </div>
    </section>
  );
}
