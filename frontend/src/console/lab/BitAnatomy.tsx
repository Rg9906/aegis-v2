import { motion } from "motion/react";
import { blockShort } from "../../lib/format";
import { FIELD_INFO } from "../constants";
import { fmtFactor, fmtValue } from "../fx";
import type { Hit } from "../types";

/** One struck float32 laid out as sign | exponent | mantissa, with the flipped bit animated. */
export default function BitAnatomy({ hits, pick, onPick }: { hits: Hit[]; pick: number; onPick: (i: number) => void }) {
  const hit = hits[pick];
  if (!hit || !hit.from_bits || !hit.to_bits) {
    return (
      <div className="anat anat-empty">
        <p className="overline">IEEE-754 anatomy</p>
        <p className="dim">Fire a strike to see exactly which of the 32 stored bits flipped and what it did to the weight.</p>
        <div className="anat-bits ghost" aria-hidden="true">
          {Array.from({ length: 32 }, (_, i) => (
            <span key={i} className={`abit ${i === 0 ? "f-sign" : i < 9 ? "f-exp" : "f-man"}`}>
              0
            </span>
          ))}
        </div>
      </div>
    );
  }
  const flipped = 31 - hit.bit; // string index of the flipped bit
  const field = hit.field ?? "any";
  return (
    <div className="anat">
      <div className="anat-head">
        <p className="overline">IEEE-754 anatomy</p>
        {hits.length > 1 && (
          <div className="anat-picker" role="group" aria-label="Choose hit">
            {hits.slice(0, 16).map((h, i) => (
              <button key={i} className={`anat-dot sv-${h.sev} ${i === pick ? "on" : ""}`} onClick={() => onPick(i)} aria-label={`Hit ${i + 1}`} />
            ))}
          </div>
        )}
      </div>
      <p className="anat-where mono">
        <span className="anat-block">{blockShort(hit.b)}</span> {hit.p}
        <span className="dim">[{hit.index}]</span> · bit {hit.bit}
      </p>
      <motion.div
        key={`${hit.p}-${hit.index}-${hit.bit}-${pick}`}
        className="anat-bits"
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: 0.012 } } }}
      >
        {hit.to_bits.split("").map((bit, i) => (
          <motion.span
            key={i}
            className={`abit ${i === 0 ? "f-sign" : i < 9 ? "f-exp" : "f-man"} ${i === flipped ? "flipped" : ""}`}
            variants={{ hidden: { opacity: 0, y: -6 }, show: { opacity: 1, y: 0 } }}
          >
            {i === flipped ? (
              <motion.span
                className="abit-flip"
                initial={{ rotateX: 0 }}
                animate={{ rotateX: 360 }}
                transition={{ delay: 0.55, duration: 0.6, ease: [0.2, 0.7, 0.1, 1] }}
              >
                {bit}
              </motion.span>
            ) : (
              bit
            )}
          </motion.span>
        ))}
      </motion.div>
      <div className="anat-legend mono">
        <span className="f-sign" title="sign">±</span>
        <span className="f-exp">exponent ×8</span>
        <span className="f-man">mantissa ×23</span>
      </div>
      <div className="anat-values">
        <div>
          <span className="dim">stored</span>
          <strong className="mono">{fmtValue(hit.from)}</strong>
        </div>
        <span className="anat-arrow" aria-hidden="true">
          →
        </span>
        <div>
          <span className="dim">after flip</span>
          <strong className={`mono sv-${hit.sev}`}>{fmtValue(hit.to)}</strong>
        </div>
        <div className={`anat-factor sv-${hit.sev}`}>
          <span className="dim">change</span>
          <strong className="mono">{fmtFactor(hit.from, hit.to)}</strong>
        </div>
      </div>
      <p className="anat-blurb dim">
        <span className={`sev-tag sv-${hit.sev}`}>{hit.sev}</span> {FIELD_INFO[field].blurb}
      </p>
    </div>
  );
}
