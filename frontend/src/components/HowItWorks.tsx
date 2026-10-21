import { motion } from "motion/react";
import SectionHead from "./SectionHead";

const STAGES = [
  {
    n: "T1 + T2",
    title: "Sense",
    body: "Hooks on every layer watch each incoming image batch. Tier 1 checks activations against envelopes calibrated on clean imagery; Tier 2 checks batch-norm statistics for drift. No labels and no ground contact needed.",
    foot: "Runs on 16 images per hour",
  },
  {
    n: "T3",
    title: "Confirm",
    body: "Behavior only says where to look. A CRC-32 of each suspect block (plus the block just upstream) is compared with its golden checksum, so only blocks whose stored bits really changed count as damaged.",
    foot: "Bit-exact, no false repairs",
  },
  {
    n: "Repair",
    title: "Restore",
    body: "Only the confirmed blocks are reloaded from a zstd-compressed golden copy in onboard flash. The rest of the model keeps running untouched. An optional 6-hour full audit catches faults too small to change behavior.",
    foot: "Layer-selective, onboard",
  },
];

export default function HowItWorks() {
  return (
    <section className="section" id="how">
      <SectionHead
        index="01"
        kicker="How it works"
        title={<>Sense. Confirm. <em>Restore only what broke.</em></>}
        lede="Scrubbing reloads the entire model on a timer whether or not anything is wrong. Aegis-v2 reloads nothing until damage is both observed and proven, and then reloads only the damaged layers."
      />
      <div className="stages">
        {STAGES.map((s, i) => (
          <motion.div
            key={s.title}
            className="panel stage"
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6, delay: i * 0.1, ease: [0.2, 0.7, 0.1, 1] }}
          >
            <div className="stage-top">
              <span className="stage-n mono">{s.n}</span>
              <span className="stage-step mono">0{i + 1}</span>
            </div>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
            <div className="stage-foot mono">{s.foot}</div>
            {i < STAGES.length - 1 && <div className="stage-arrow" aria-hidden="true" />}
          </motion.div>
        ))}
      </div>
    </section>
  );
}
