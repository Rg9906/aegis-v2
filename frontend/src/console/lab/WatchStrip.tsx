import { motion } from "motion/react";
import { CLASS_SHORT } from "../constants";
import { chipUrl } from "../engine";
import type { WatchChip } from "../types";

/** One real chip per class: what the live payload says now vs the golden model. */
export default function WatchStrip({ watch, measuring }: { watch: WatchChip[] | null; measuring: boolean }) {
  return (
    <div className={`watch ${measuring ? "measuring" : ""}`}>
      {(watch ?? []).map((w) => {
        const ok = w.pred === w.label;
        return (
          <figure key={w.index} className={`watch-chip ${ok ? "ok" : "bad"}`}>
            <div className="watch-img">
              <img src={chipUrl(w.index, 2)} alt={CLASS_SHORT[w.label]} width={64} height={64} loading="lazy" />
              <span className="watch-scan" aria-hidden="true" />
            </div>
            <figcaption>
              <span className="watch-truth dim">{CLASS_SHORT[w.label]}</span>
              <motion.span
                key={`${w.index}-${w.pred}`}
                className="watch-pred mono"
                initial={{ rotateX: 90, opacity: 0 }}
                animate={{ rotateX: 0, opacity: 1 }}
                transition={{ duration: 0.4 }}
              >
                {ok ? "✓ " : "✗ "}
                {CLASS_SHORT[w.pred] ?? "—"}
              </motion.span>
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}
