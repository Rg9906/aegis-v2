import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { blockShort, missionClock } from "../../lib/format";
import { useTheme } from "../../lib/useMissionClock";
import { useEngine } from "../engine";
import { STRATEGY_COLOR } from "../constants";
import { fmtValue, pct1 } from "../fx";
import type { MissionConfig, MissionState, StrategyKey } from "../types";
import LiveMap from "./LiveMap";
import StrategyCard from "./StrategyCard";
import MissionChart from "./MissionChart";

const DURATIONS = [
  { hours: 6, step: 30, label: "6 h" },
  { hours: 12, step: 60, label: "12 h" },
  { hours: 24, step: 120, label: "24 h" },
  { hours: 72, step: 240, label: "3 days" },
];
const FLUX = [
  { x: 1, label: "1×", hint: "Calibrated to Alsat-1 flight data (~145 flips/day). Damage is rare in a short run." },
  { x: 10, label: "10×", hint: "Accelerated for demonstration: SAA passes reliably cause damaging hits." },
  { x: 30, label: "30×", hint: "Storm: heavy, sustained bombardment. Strongly accelerated; disclose when presenting." },
];
const PROBES = [
  { ppc: 2, label: "20 chips", hint: "fastest" },
  { ppc: 3, label: "30 chips", hint: "balanced" },
  { ppc: 5, label: "50 chips", hint: "finer accuracy" },
];
const SCRUBS = [3, 6, 12];

export default function Mission() {
  const { mission, snapshot, run, busy } = useEngine();
  const { theme } = useTheme();
  const [dur, setDur] = useState(1);
  const [flux, setFlux] = useState(10);
  const [ppc, setPpc] = useState(2);
  const [scrub, setScrub] = useState(6);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1000));

  const running = !!mission?.running;
  const cfg: MissionConfig = { hours: DURATIONS[dur].hours, step_minutes: DURATIONS[dur].step, flux, scrub_hours: scrub, audit_hours: 6, probe_per_class: ppc, seed };
  const steps = (cfg.hours * 60) / cfg.step_minutes;
  // Measured pace of the last mission if any, else ~1.6 s per 10 probe chips per step
  const lastPace = mission?.steps.length ? mission.steps.reduce((a, s) => a + s.wall_s, 0) / mission.steps.length / (mission.config.probe_per_class / 2) : null;
  const eta = steps * (lastPace ?? 1.8) * (ppc / 2);

  const launch = () => run("Launch mission", "mission/start", cfg);
  const abort = () => run("Abort mission", "mission/stop", {});

  return (
    <div className="mission">
      <section className="m-config panel-glass" aria-label="Mission setup">
        <p className="overline">Mission profile</p>
        <div className="ctl">
          <span className="ctl-label">Duration</span>
          <div className="seg seg-sm" role="group" aria-label="Duration">
            {DURATIONS.map((d, i) => (
              <button key={d.label} className={dur === i ? "on" : ""} onClick={() => setDur(i)} disabled={running}>
                {d.label}
              </button>
            ))}
          </div>
          <p className="ctl-hint dim">
            {steps} steps of {cfg.step_minutes} min on the real Sentinel-2A track
          </p>
        </div>
        <div className="ctl">
          <span className="ctl-label">Radiation intensity</span>
          <div className="seg seg-sm" role="group" aria-label="Radiation intensity">
            {FLUX.map((f) => (
              <button key={f.x} className={flux === f.x ? "on" : ""} onClick={() => setFlux(f.x)} disabled={running}>
                {f.label}
              </button>
            ))}
          </div>
          <p className={`ctl-hint ${flux > 1 ? "warn" : "dim"}`}>{FLUX.find((f) => f.x === flux)?.hint}</p>
        </div>
        <div className="ctl">
          <span className="ctl-label">Scrubbing baseline every</span>
          <div className="seg seg-sm" role="group" aria-label="Scrub interval">
            {SCRUBS.map((h) => (
              <button key={h} className={scrub === h ? "on" : ""} onClick={() => setScrub(h)} disabled={running}>
                {h} h
              </button>
            ))}
          </div>
        </div>
        <div className="ctl">
          <span className="ctl-label">Accuracy probe</span>
          <div className="seg seg-sm" role="group" aria-label="Probe size">
            {PROBES.map((p) => (
              <button key={p.ppc} className={ppc === p.ppc ? "on" : ""} onClick={() => setPpc(p.ppc)} disabled={running} title={p.hint}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <label className="ctl ctl-row">
          <span className="ctl-label">Radiation seed</span>
          <span className="seed">
            <input type="number" min={0} max={99999} value={seed} onChange={(e) => setSeed(Math.max(0, Math.min(99999, Number(e.target.value) || 0)))} disabled={running} />
            <button className="icon-btn" onClick={() => setSeed(Math.floor(Math.random() * 100000))} disabled={running} aria-label="Random seed" title="Random seed">
              ⚄
            </button>
          </span>
        </label>
        {running ? (
          <button className="fire fire-abort" onClick={abort} disabled={!!mission?.stopping}>
            <span className="fire-label">{mission?.stopping ? "Stopping after this step…" : "Abort mission"}</span>
          </button>
        ) : (
          <motion.button className="fire fire-go" onClick={launch} disabled={!!busy || snapshot?.status !== "ready"} whileTap={{ scale: 0.97 }}>
            <span className="fire-glow" aria-hidden="true" />
            <span className="fire-label">Launch mission</span>
            <span className="fire-key mono">≈{eta < 90 ? `${Math.round(eta)} s` : `${(eta / 60).toFixed(1)} min`}</span>
          </motion.button>
        )}
        <p className="ctl-hint dim">
          Four replicas of the {snapshot?.arch_label} payload receive the identical seeded fault stream; every number below is computed
          live by the engine.
        </p>
      </section>

      <section className="m-stage">
        {mission ? <MissionView mission={mission} theme={theme} /> : <MissionIdle />}
      </section>
    </div>
  );
}

function MissionIdle() {
  return (
    <div className="m-idle panel-glass">
      <div className="boot-orbit spinning" aria-hidden="true">
        <span className="boot-sat" />
      </div>
      <h2>
        Fly the payload through the <em>South Atlantic Anomaly.</em>
      </h2>
      <p className="dim">
        Launch a mission: the satellite follows the real Sentinel-2A orbit (SGP4), radiation strikes arrive as a Poisson process
        integrated minute by minute over the IGRF field, and four protection strategies run side by side on identical faults.
      </p>
    </div>
  );
}

function MissionView({ mission, theme }: { mission: MissionState; theme: string }) {
  const last = mission.steps[mission.steps.length - 1];
  const progress = mission.steps.length / mission.config.total_steps;
  const maxMb = Math.max(1, ...mission.strategies.map((s) => last?.strategies[s.key].mb ?? 0));
  const flips = mission.steps.reduce((a, s) => a + s.flips, 0);
  const msb = mission.steps.reduce((a, s) => a + s.msb, 0);
  const feed = useMemo(
    () =>
      mission.steps
        .slice(-6)
        .reverse()
        .flatMap((s) => s.hits.filter((h) => h.sev !== "minor").map((h) => ({ ...h, step: s.step, t: h.t_min ?? s.t_min })))
        .slice(0, 14),
    [mission.steps],
  );

  // Elapsed wall clock while running
  const [, tick] = useState(0);
  useEffect(() => {
    if (!mission.running) return;
    const id = window.setInterval(() => tick((n) => n + 1), 500);
    return () => window.clearInterval(id);
  }, [mission.running]);
  const wall = mission.running ? Date.now() / 1000 - mission.started : mission.summary?.wall_s ?? 0;

  return (
    <>
      <div className="m-map panel-glass">
        <LiveMap mission={mission} theme={theme} />
        <div className="m-hud">
          <div className="hud-block">
            <span className="hud-label">Mission #{mission.id}</span>
            <span className="hud-value mono">{missionClock(last?.t_min ?? 0)}</span>
          </div>
          <div className="hud-block">
            <span className="hud-label">Step</span>
            <span className="hud-value mono">
              {mission.steps.length}/{mission.config.total_steps}
            </span>
          </div>
          <div className="hud-block">
            <span className="hud-label">Bit flips</span>
            <span className="hud-value mono">
              {flips}
              <span className="hud-crit"> · {msb} exp-MSB</span>
            </span>
          </div>
          <div className="hud-block">
            <span className="hud-label">Radiation</span>
            <span className={`hud-value mono ${last?.in_saa ? "hud-saa" : ""}`}>{last?.in_saa ? `SAA · ${last.flux_x.toFixed(0)}×` : "background"}</span>
          </div>
          <div className="hud-block">
            <span className="hud-label">Wall clock</span>
            <span className="hud-value mono">{wall.toFixed(0)} s</span>
          </div>
        </div>
        <div className="m-progress" aria-hidden="true">
          <motion.span animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.5 }} />
        </div>
        {mission.config.flux > 1 && <span className="m-badge">Radiation accelerated {mission.config.flux}× · {mission.expected_flips_per_day.toFixed(0)} flips/day</span>}
        {mission.running && mission.steps.length === 0 && <span className="m-badge m-badge-center">Preparing replicas and first orbit step…</span>}
      </div>

      <AnimatePresence>{mission.summary && !mission.running && <Debrief mission={mission} />}</AnimatePresence>
      {mission.error && <p className="m-error">Mission failed: {mission.error}</p>}

      <div className="strats">
        {mission.strategies.map((s) => (
          <StrategyCard key={s.key} k={s.key} label={s.label} mission={mission} maxMb={maxMb} />
        ))}
      </div>

      <div className="m-bottom">
        <div className="panel-glass">
          <div className="block-head">
            <h3>Accuracy on {mission.probe_size} held-out chips</h3>
            <span className="hint">{mission.arch_label} · clean {pct1(mission.clean_acc)}</span>
          </div>
          <MissionChart mission={mission} />
        </div>
        <div className="panel-glass m-feed">
          <div className="block-head">
            <h3>Damaging hits</h3>
            <span className="hint">moderate + critical</span>
          </div>
          <ul>
            <AnimatePresence initial={false}>
              {feed.map((h) => (
                <motion.li
                  key={`${h.step}-${h.p}-${h.bit}-${String(h.from)}`}
                  layout
                  initial={{ opacity: 0, x: -14 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className={`feed-${h.sev}`}
                >
                  <span className="mono feed-t">{missionClock(h.t)}</span>
                  <span className="mono feed-b">{blockShort(h.b)}</span>
                  <span className="mono feed-v">
                    bit {h.bit}: {fmtValue(h.from)} → {fmtValue(h.to)}
                  </span>
                </motion.li>
              ))}
            </AnimatePresence>
            {feed.length === 0 && <li className="dim">No damaging hits yet: most flips land in harmless mantissa bits.</li>}
          </ul>
        </div>
      </div>
    </>
  );
}

function Debrief({ mission }: { mission: MissionState }) {
  const s = mission.summary!;
  const order: StrategyKey[] = ["unmitigated", "scrub", "aegis", "audit"];
  const aegis = s.strategies.aegis;
  const scrub = s.strategies.scrub;
  const verdict =
    aegis.mean_acc != null && scrub.mean_acc != null
      ? `Aegis-v2 held ${pct1(aegis.mean_acc)} mean accuracy with ${aegis.mb.toFixed(1)} MB of repair traffic; blind scrubbing every ${mission.config.scrub_hours} h held ${pct1(scrub.mean_acc)} with ${scrub.mb.toFixed(1)} MB.`
      : "";
  return (
    <motion.div className="debrief panel-glass" initial={{ opacity: 0, y: 18, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0 }}>
      <div className="debrief-head">
        <p className="overline">{s.stopped ? "Mission aborted" : "Mission debrief"}</p>
        <h3>
          {s.hours.toFixed(0)} h in orbit · {s.flips} bit flips · {s.msb} exponent-MSB hits
        </h3>
        <p className="dim">{verdict}</p>
      </div>
      <table className="debrief-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Mean acc.</th>
            <th>Worst</th>
            <th>Final</th>
            <th>Traffic</th>
            <th>Repairs</th>
          </tr>
        </thead>
        <tbody>
          {order.map((k) => {
            const r = s.strategies[k];
            return (
              <tr key={k}>
                <td>
                  <i className="swatch" style={{ background: STRATEGY_COLOR[k] }} /> {r.label}
                </td>
                <td className="mono">{pct1(r.mean_acc)}</td>
                <td className="mono">{pct1(r.min_acc)}</td>
                <td className="mono">{pct1(r.final_acc)}</td>
                <td className="mono">{r.mb.toFixed(1)} MB</td>
                <td className="mono">{r.repairs}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="debrief-note dim">
        One seed, {mission.probe_size} probe chips{mission.config.flux > 1 ? `, radiation accelerated ${mission.config.flux}×` : ""}. The recorded
        multi-seed campaigns in the Briefing are the evidence; this run is a live demonstration. Computed in {s.wall_s.toFixed(0)} s.
      </p>
    </motion.div>
  );
}
