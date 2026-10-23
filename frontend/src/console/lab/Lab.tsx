import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { blockShort } from "../../lib/format";
import { useEngine } from "../engine";
import { AnimatedNumber, mbStr } from "../fx";
import { FIELD_INFO } from "../constants";
import type { BitField, Diagnosis, Hit, Measure, Repair } from "../types";
import NetworkMap, { type NodeFx } from "./NetworkMap";
import BitAnatomy from "./BitAnatomy";
import DetectionFunnel, { type FunnelPhase } from "./DetectionFunnel";
import WatchStrip from "./WatchStrip";

type Phase = "idle" | "strike" | "measuring" | "impact" | "scanning" | "diagnosed" | "repairing" | "repaired" | "recovered";

// Minimum on-screen time per stage so a fast engine still reads as a sequence
const DWELL: Record<string, number> = { strike: 1100, impact: 1000, diagnosis: 1500, repair: 1300, recovered: 700 };
const COUNTS = [1, 4, 16];
const EMPTY = () => new Set<string>();

export default function Lab() {
  const { snapshot, run, busy, subscribe } = useEngine();
  const blocks = snapshot?.blocks ?? [];
  const armed = !!snapshot?.armed;

  const [target, setTarget] = useState<string>("layer3_block1");
  const [field, setField] = useState<BitField>("exponent_msb");
  const [count, setCount] = useState(1);
  const [phase, setPhase] = useState<Phase>("idle");
  const [hits, setHits] = useState<Hit[]>([]);
  const [pick, setPick] = useState(0);
  const [measure, setMeasure] = useState<Measure | null>(null);
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null);
  const [repair, setRepair] = useState<Repair | null>(null);
  const [fx, setFx] = useState<NodeFx>({
    hit: EMPTY(), suspects: EMPTY(), confirmed: EMPTY(), restoring: EMPTY(), restored: EMPTY(), flowing: false, strikeSeq: 0,
  });

  // Fall back to a valid target when the architecture changes
  useEffect(() => {
    if (blocks.length && target !== "random" && !blocks.some((b) => b.id === target)) {
      setTarget(blocks[Math.min(5, blocks.length - 1)].id);
    }
  }, [blocks, target]);

  // ---- choreography: engine events are queued and replayed with a minimum dwell ------------
  const queue = useRef<{ type: string; data: unknown }[]>([]);
  const timer = useRef<number | null>(null);
  const armedRef = useRef(armed);
  armedRef.current = armed;
  const fadeTimer = useRef<number | null>(null);

  const apply = useCallback((type: string, data: unknown) => {
    if (fadeTimer.current) window.clearTimeout(fadeTimer.current);
    switch (type) {
      case "strike": {
        const h = (data as { hits: Hit[] }).hits;
        setHits(h);
        setPick(Math.max(0, h.findIndex((x) => x.sev === "critical")));
        setDiagnosis(null);
        setRepair(null);
        setPhase("strike");
        setFx((f) => ({ ...f, hit: new Set(h.map((x) => x.b)), suspects: EMPTY(), confirmed: EMPTY(), restoring: EMPTY(), restored: EMPTY(), flowing: false, strikeSeq: f.strikeSeq + 1 }));
        break;
      }
      case "impact":
        setMeasure(data as Measure);
        setPhase("impact");
        setFx((f) => ({ ...f, flowing: false }));
        break;
      case "diagnosis": {
        const d = data as Diagnosis;
        setDiagnosis(d);
        setRepair(null);
        setPhase("diagnosed");
        setFx((f) => ({ ...f, suspects: new Set(d.suspects), confirmed: new Set(d.confirmed), restored: EMPTY(), flowing: false }));
        break;
      }
      case "repair": {
        const r = data as Repair;
        setRepair(r);
        setPhase("repairing");
        setFx((f) => ({ ...f, restoring: new Set(r.blocks), suspects: EMPTY() }));
        window.setTimeout(() => {
          setPhase((p) => (p === "repairing" ? "repaired" : p));
          setFx((f) => ({ ...f, restoring: EMPTY(), confirmed: EMPTY(), restored: new Set(r.blocks), hit: EMPTY() }));
        }, 900);
        break;
      }
      case "recovered":
        setMeasure(data as Measure);
        setPhase("recovered");
        fadeTimer.current = window.setTimeout(() => setFx((f) => ({ ...f, restored: EMPTY() })), 4000);
        break;
    }
  }, []);

  const pump = useCallback(() => {
    if (timer.current !== null) return;
    const next = queue.current.shift();
    if (!next) return;
    apply(next.type, next.data);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      // Waiting on the engine for the next stage: show what it is computing
      if (!queue.current.length) {
        if (next.type === "strike") {
          setPhase("measuring");
          setFx((f) => ({ ...f, flowing: true }));
        }
        if (next.type === "impact" && armedRef.current) {
          setPhase("scanning");
          setFx((f) => ({ ...f, flowing: true }));
        }
      }
      pump();
    }, DWELL[next.type] ?? 600);
  }, [apply]);

  useEffect(() => {
    const types = ["strike", "impact", "diagnosis", "repair", "recovered"];
    const offs = types.map((t) =>
      subscribe(t, (ev) => {
        queue.current.push({ type: t, data: ev.data });
        pump();
      }),
    );
    return () => {
      offs.forEach((o) => o());
      if (timer.current) window.clearTimeout(timer.current);
      if (fadeTimer.current) window.clearTimeout(fadeTimer.current);
    };
  }, [subscribe, pump]);

  // ---- commands ------------------------------------------------------------------------------
  const disabled = !!busy || !!snapshot?.mission_running;
  const fire = useCallback(() => {
    if (disabled) return;
    void run("Strike", "strike", { block: target, field, count });
  }, [disabled, run, target, field, count]);

  const manual = (name: string, path: string, body: unknown, optimistic?: Phase) => {
    if (disabled) return;
    if (optimistic) {
      setPhase(optimistic);
      setFx((f) => ({ ...f, flowing: optimistic === "scanning" }));
    }
    void run(name, path, body).then((res) => {
      setFx((f) => ({ ...f, flowing: false }));
      if (path === "reset" && res) {
        queue.current = [];
        setHits([]);
        setMeasure(null);
        setDiagnosis(null);
        setRepair(null);
        setPhase("idle");
        setFx((f) => ({ ...f, hit: EMPTY(), suspects: EMPTY(), confirmed: EMPTY(), restoring: EMPTY(), restored: EMPTY() }));
      } else if (optimistic && !res) {
        setPhase("idle"); // command refused: drop the optimistic scanning state
      }
    });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      if (e.repeat || e.metaKey || e.ctrlKey || (el && ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName))) return;
      if (e.key.toLowerCase() === "f") fire();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fire]);

  // ---- derived -------------------------------------------------------------------------------
  const funnelPhase: FunnelPhase =
    phase === "scanning" ? "scanning" : phase === "diagnosed" ? "diagnosed" : phase === "repairing" ? "repairing" : phase === "repaired" || phase === "recovered" ? "repaired" : diagnosis ? "diagnosed" : "idle";
  const acc = measure?.accuracy ?? snapshot?.live_acc ?? 0;
  const clean = snapshot?.clean_acc ?? 0;
  const corrupted = snapshot?.corrupted ?? [];
  const watch = measure?.watch ?? snapshot?.watch ?? null;
  const integrity = corrupted.length === 0 ? "nominal" : diagnosis?.silent.length || (diagnosis?.healthy && corrupted.length) ? "silent" : "corrupt";
  const status = useMemo(() => {
    const s: Record<Phase, string> = {
      idle: "Standing by. Pick a block and a bit field, then fire.",
      strike: `Particle strike: ${hits.length} bit${hits.length === 1 ? "" : "s"} flipped in memory`,
      measuring: `Running ${snapshot?.probe_size ?? ""} real chips through the corrupted payload…`,
      impact: `Accuracy ${(acc * 100).toFixed(1)}% after the strike`,
      scanning: "Aegis observes the next unlabeled batch: T1 + T2 hooks, then CRC on suspects…",
      diagnosed: diagnosis
        ? diagnosis.confirmed.length
          ? `Alarm confirmed bit-exact in ${diagnosis.confirmed.map(blockShort).join(", ")}`
          : diagnosis.false_alarm
            ? "Behavioral alarm, but CRC finds nothing: false alarm absorbed"
            : diagnosis.silent.length
              ? "No alarm: corruption is below the behavioral threshold"
              : "Healthy: every envelope and BN layer within bounds"
        : "",
      repairing: `Restoring ${repair?.blocks.map(blockShort).join(", ") ?? ""} from the golden store…`,
      repaired: `Restored ${repair ? mbStr(repair.bytes) : ""} MB, CRC verified`,
      recovered: `Recovered: accuracy back to ${(acc * 100).toFixed(1)}%`,
    };
    return s[phase];
  }, [phase, hits.length, acc, diagnosis, repair, snapshot?.probe_size]);

  return (
    <div className="lab">
      <section className="lab-controls panel-glass" aria-label="Strike controls">
        <p className="overline">Radiation strike</p>
        <label className="ctl">
          <span className="ctl-label">Target block</span>
          <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={disabled}>
            <option value="random">Random (weighted by memory size)</option>
            {blocks.map((b) => (
              <option key={b.id} value={b.id}>
                {blockShort(b.id)} · {b.id} · {(b.bytes / 1e6).toFixed(2)} MB
              </option>
            ))}
          </select>
        </label>
        <div className="ctl">
          <span className="ctl-label">Bit field</span>
          <div className="field-grid" role="radiogroup" aria-label="Bit field">
            {(Object.keys(FIELD_INFO) as BitField[]).map((f) => (
              <button key={f} role="radio" aria-checked={field === f} className={`field-btn ${field === f ? "on" : ""} f-${f}`} onClick={() => setField(f)} disabled={disabled}>
                <span>{FIELD_INFO[f].label}</span>
                <span className="mono dim">{FIELD_INFO[f].tag}</span>
              </button>
            ))}
          </div>
          <p className="ctl-hint dim">{FIELD_INFO[field].blurb}</p>
        </div>
        <div className="ctl ctl-row">
          <span className="ctl-label">Bits per strike</span>
          <div className="seg seg-sm" role="group" aria-label="Bits per strike">
            {COUNTS.map((c) => (
              <button key={c} className={count === c ? "on" : ""} onClick={() => setCount(c)} disabled={disabled}>
                {c}
              </button>
            ))}
          </div>
        </div>
        <motion.button className="fire" onClick={fire} disabled={disabled} whileTap={{ scale: 0.97 }} aria-keyshortcuts="F">
          <span className="fire-glow" aria-hidden="true" />
          <span className="fire-label">{busy === "Strike" ? "Strike in progress…" : "Fire strike"}</span>
          <span className="fire-key mono" aria-hidden="true">
            F
          </span>
        </motion.button>

        <div className={`autopilot ${armed ? "on" : ""}`}>
          <div>
            <span className="ctl-label">Aegis autopilot</span>
            <p className="dim">{armed ? "Alarms trigger an automatic CRC-gated repair." : "Off: you diagnose and repair by hand."}</p>
          </div>
          <button
            className="toggle"
            role="switch"
            aria-checked={armed}
            aria-label="Aegis autopilot"
            onClick={() => run("Toggle autopilot", "arm", { armed: !armed })}
            disabled={disabled}
          >
            <motion.span layout className="toggle-knob" transition={{ type: "spring", stiffness: 500, damping: 32 }} />
          </button>
        </div>

        <p className="overline ops-title">Manual operations</p>
        <div className="ops">
          <button className="btn" disabled={disabled} onClick={() => manual("Health check", "diagnose", { full_crc: false }, "scanning")} title="Run one monitored batch (T1 + T2, CRC on suspects)">
            Health check
          </button>
          <button className="btn" disabled={disabled} onClick={() => manual("Aegis repair", "repair", { mode: "aegis" }, "scanning")} title="Diagnose, then restore only CRC-confirmed blocks">
            Aegis repair
          </button>
          <button className="btn" disabled={disabled} onClick={() => manual("CRC audit", "repair", { mode: "audit" })} title="CRC-scan every block; restore any mismatch (catches silent faults)">
            CRC audit
          </button>
          <button className="btn" disabled={disabled} onClick={() => manual("Full scrub", "repair", { mode: "scrub" })} title="Blind full reload: the scrubbing baseline">
            Full scrub
          </button>
          <button className="btn btn-ghost" disabled={disabled} onClick={() => manual("Reset", "reset", {})} title="Restore golden weights and clear counters">
            Reset payload
          </button>
        </div>
        {snapshot?.mission_running && <p className="ctl-hint warn">A live mission is running: the lab is paused until it ends.</p>}
      </section>

      <section className="lab-stage" aria-label="Payload">
        <div className="vitals">
          <div className="vital vital-acc">
            <Gauge value={acc} clean={clean} />
            <div>
              <span className="vital-label">Payload accuracy</span>
              <span className="vital-value">
                <AnimatedNumber value={acc * 100} suffix="%" />
              </span>
              <span className="vital-sub mono">clean {(clean * 100).toFixed(1)}% · {snapshot?.probe_size} probe chips</span>
            </div>
          </div>
          <div className={`vital vital-int int-${integrity}`}>
            <span className="vital-label">Memory integrity</span>
            <span className="vital-value">{integrity === "nominal" ? "Nominal" : integrity === "silent" ? "Silent faults" : "Corrupted"}</span>
            <span className="vital-sub mono">
              {corrupted.length ? `CRC ✗ ${corrupted.map(blockShort).join(" ")}` : `all ${blocks.length} blocks match golden CRC`}
            </span>
          </div>
          <div className="vital">
            <span className="vital-label">Repair traffic</span>
            <span className="vital-value">
              <AnimatedNumber value={(snapshot?.bytes_restored ?? 0) / 1e6} decimals={2} suffix=" MB" />
            </span>
            <span className="vital-sub mono">full reload = {mbStr(snapshot?.golden_bytes ?? 0)} MB</span>
          </div>
          <div className="vital">
            <span className="vital-label">Strikes · repairs</span>
            <span className="vital-value tabular">
              {snapshot?.strikes ?? 0} · {snapshot?.repairs ?? 0}
            </span>
            <span className="vital-sub mono">{snapshot?.arch_label} · {blocks.length} restorable blocks</span>
          </div>
        </div>

        <div className={`stage-status phase-${phase}`} role="status" aria-live="polite">
          <span className="stage-dot" aria-hidden="true" />
          <AnimatePresence mode="wait">
            <motion.span key={status} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
              {status}
            </motion.span>
          </AnimatePresence>
        </div>

        <div className="panel-glass map-panel">
          <div className="block-head">
            <h3>Payload memory · {snapshot?.arch_label}</h3>
            <span className="hint">Click a block to target it</span>
          </div>
          <NetworkMap blocks={blocks} fx={fx} selected={target} onSelect={setTarget} disabled={disabled} />
        </div>

        <div className="lab-grid">
          <div className="panel-glass">
            <BitAnatomy hits={hits} pick={pick} onPick={setPick} />
          </div>
          <div className="panel-glass">
            <p className="overline">Detection pipeline</p>
            <DetectionFunnel phase={funnelPhase} diagnosis={diagnosis} repair={repair} armed={armed} />
          </div>
        </div>

        <div className="panel-glass">
          <div className="block-head">
            <h3>Live predictions · one real chip per class</h3>
            <span className="hint">{measure ? `measured in ${(measure.ms / 1000).toFixed(1)} s` : "golden baseline"}</span>
          </div>
          <WatchStrip watch={watch} measuring={phase === "measuring"} />
        </div>
      </section>
    </div>
  );
}

/** Radial accuracy gauge with a tick at the clean baseline. */
function Gauge({ value, clean }: { value: number; clean: number }) {
  const R = 30;
  const C = 2 * Math.PI * R;
  const tone = value >= clean - 0.05 ? "good" : value >= 0.4 ? "warn" : "crit";
  const tick = clean * 2 * Math.PI - Math.PI / 2;
  return (
    <svg className={`acc-ring acc-ring-${tone}`} width="76" height="76" viewBox="0 0 76 76" aria-hidden="true">
      <circle cx="38" cy="38" r={R} className="acc-ring-track" />
      <motion.circle
        cx="38"
        cy="38"
        r={R}
        className="acc-ring-arc"
        strokeDasharray={C}
        initial={false}
        animate={{ strokeDashoffset: C * (1 - Math.max(0, Math.min(1, value))) }}
        transition={{ duration: 0.9, ease: [0.2, 0.7, 0.1, 1] }}
        transform="rotate(-90 38 38)"
      />
      <line x1={38 + (R - 6) * Math.cos(tick)} y1={38 + (R - 6) * Math.sin(tick)} x2={38 + (R + 6) * Math.cos(tick)} y2={38 + (R + 6) * Math.sin(tick)} className="acc-ring-tick" />
    </svg>
  );
}
