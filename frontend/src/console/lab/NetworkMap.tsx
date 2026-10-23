import { AnimatePresence, motion } from "motion/react";
import { blockShort } from "../../lib/format";
import type { BlockInfo, Hit } from "../types";

export interface NodeFx {
  hit: Set<string>; // struck in the current sequence
  suspects: Set<string>; // T1/T2 implicated
  confirmed: Set<string>; // T3 CRC mismatch
  restoring: Set<string>; // being reloaded right now
  restored: Set<string>; // just reloaded and verified
  flowing: boolean; // inference running through the network
  strikeSeq: number; // bumps on every strike to replay the beam
}

interface Props {
  blocks: BlockInfo[];
  fx: NodeFx;
  selected: string;
  onSelect: (id: string) => void;
  disabled: boolean;
}

const CELLS = 28;

/** Memory cell lit by a hit: stable position derived from the struck element index. */
const cellOf = (h: Hit) => (Math.imul((h.index ?? h.bit * 7919) | 0, 2654435761 | 0) >>> 0) % CELLS;

/**
 * The payload as a data path, input -> blocks -> logits. Node width follows sqrt(memory size);
 * each node shows its memory cells, live CRC state and the detection/repair overlays.
 */
export default function NetworkMap({ blocks, fx, selected, onSelect, disabled }: Props) {
  const maxBytes = Math.max(...blocks.map((b) => b.bytes));
  return (
    <div className={`netmap ${fx.flowing ? "flowing" : ""}`} role="group" aria-label="Payload memory map">
      <div className="net-cap net-in">
        <span className="net-cap-icon" aria-hidden="true">
          ◱
        </span>
        <span>
          Sentinel-2
          <br />
          chips
        </span>
      </div>
      {blocks.map((b) => {
        const state = fx.restoring.has(b.id)
          ? "restoring"
          : fx.confirmed.has(b.id)
            ? "confirmed"
            : fx.restored.has(b.id)
              ? "restored"
              : fx.suspects.has(b.id)
                ? "suspect"
                : !b.crc_ok
                  ? "corrupt"
                  : "healthy";
        const grow = 0.55 + 2.2 * Math.sqrt(b.bytes / maxBytes);
        const lit = new Map<number, Hit["sev"]>();
        for (const h of b.recent) lit.set(cellOf(h), h.sev);
        const struck = fx.hit.has(b.id);
        return (
          <button
            key={b.id}
            className={`net-node state-${state} ${selected === b.id ? "selected" : ""}`}
            style={{ flexGrow: grow }}
            onClick={() => onSelect(selected === b.id ? "random" : b.id)}
            disabled={disabled}
            aria-pressed={selected === b.id}
            title={`${b.id} · ${(b.bytes / 1e6).toFixed(2)} MB · ${b.params.toLocaleString()} params${b.faults ? ` · ${b.faults} active fault(s)` : ""}\nClick to target this block`}
          >
            <AnimatePresence>
              {struck && (
                <motion.span
                  key={`beam-${fx.strikeSeq}`}
                  className="net-beam"
                  initial={{ scaleY: 0, opacity: 1 }}
                  animate={{ scaleY: [0, 1, 1], opacity: [1, 1, 0] }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.75, times: [0, 0.35, 1], ease: "easeOut" }}
                  aria-hidden="true"
                />
              )}
            </AnimatePresence>
            {struck && (
              <motion.span
                key={`ring-${fx.strikeSeq}`}
                className="net-ring"
                initial={{ scale: 0.2, opacity: 0.9 }}
                animate={{ scale: 2.4, opacity: 0 }}
                transition={{ duration: 0.9, delay: 0.25, ease: "easeOut" }}
                aria-hidden="true"
              />
            )}
            <motion.span
              className="net-node-inner"
              animate={struck ? { x: [0, -5, 5, -3, 3, 0] } : { x: 0 }}
              transition={{ duration: 0.45, delay: 0.25 }}
            >
              <span className="net-node-head">
                <span className="net-name mono">{blockShort(b.id)}</span>
                <span className={`net-crc ${b.crc_ok ? "ok" : "bad"}`} title={b.crc_ok ? "CRC matches golden" : "CRC mismatch"}>
                  {b.crc_ok ? "✓" : "✗"}
                </span>
              </span>
              <span className="net-cells" aria-hidden="true">
                {Array.from({ length: CELLS }, (_, i) => (
                  <i key={i} className={lit.has(i) ? `lit-${lit.get(i)}` : ""} />
                ))}
              </span>
              <span className="net-node-foot mono">
                <span>{b.bytes >= 1e6 ? `${(b.bytes / 1e6).toFixed(1)} MB` : `${(b.bytes / 1e3).toFixed(0)} KB`}</span>
                {b.faults > 0 && <span className={`net-faults sv-${b.worst}`}>{b.faults}⚡</span>}
              </span>
              {state === "restoring" && <span className="net-sweep" aria-hidden="true" />}
            </motion.span>
          </button>
        );
      })}
      <div className="net-cap net-out">
        <span className="net-cap-icon" aria-hidden="true">
          ▤
        </span>
        <span>
          10-class
          <br />
          land cover
        </span>
      </div>
    </div>
  );
}
