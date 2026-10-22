import { useEffect, useMemo, useRef, useState } from "react";
import type { Campaign, StrategyAttr } from "../../lib/types";
import { seedKeys } from "../../lib/data";
import { SPEEDS, useMissionClock } from "../../lib/useMissionClock";
import GroundMap from "./GroundMap";
import MemoryMap from "./MemoryMap";
import AccuracyChart from "./AccuracyChart";
import Timeline from "./Timeline";
import Telemetry from "./Telemetry";
import SectionHead from "../SectionHead";

export default function Replay({ campaign, theme }: { campaign: Campaign; theme: string }) {
  const { steps, step_minutes } = campaign.setup;
  const seeds = seedKeys(campaign);
  // Default to the most turbulent seed (most exponent-MSB hits): the most to watch
  const defaultSeed = useMemo(
    () =>
      seeds.reduce((best, k) =>
        campaign.seeds[k].summary.exponent_msb_flips_total > campaign.seeds[best].summary.exponent_msb_flips_total ? k : best,
      ),
    [campaign, seeds],
  );
  const [seed, setSeed] = useState(defaultSeed);
  const [chartSeed, setChartSeed] = useState<"follow" | "mean">("follow");
  const [hidden, setHidden] = useState<Set<StrategyAttr>>(new Set());
  const clock = useMissionClock(steps * step_minutes);
  const step = Math.min(steps - 1, Math.floor(clock.minutes / step_minutes));
  const run = campaign.seeds[seed];

  useEffect(() => setSeed(defaultSeed), [defaultSeed]);

  // Keyboard handler registered once; it reads the latest toggle through a ref
  const toggleRef = useRef(clock.toggle);
  toggleRef.current = clock.toggle;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      if (e.code !== "Space" || (el && ["INPUT", "BUTTON", "SELECT", "TEXTAREA"].includes(el.tagName))) return;
      const sec = document.getElementById("replay")!.getBoundingClientRect();
      if (sec.top > window.innerHeight || sec.bottom < 0) return; // only while the replay is on screen
      e.preventDefault();
      toggleRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const toggle = (a: StrategyAttr) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(a)) n.delete(a);
      else n.add(a);
      return n;
    });

  return (
    <section className="section" id="replay">
      <SectionHead
        index="03"
        kicker="Mission replay"
        title={<>Watch radiation <em>strike</em>, and the model recover.</>}
        lede="Every bit flip below is the exact one the campaign injected, replayed from its seed. The satellite flies the real Sentinel-2A track; the shaded region is where Earth's magnetic field is weakest and the radiation strongest."
      />

      <div className="panel replay-panel">
        <div className="replay-controls">
          <button className="btn btn-primary btn-play" onClick={clock.toggle} aria-label={clock.playing ? "Pause" : "Play"}>
            {clock.playing ? (
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor" /><rect x="8.5" y="1" width="3.5" height="12" rx="1" fill="currentColor" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11a.6.6 0 0 0 .9.5l9-5.5a.6.6 0 0 0 0-1l-9-5.5A.6.6 0 0 0 3 1.5Z" fill="currentColor" /></svg>
            )}
            {clock.playing ? "Pause" : clock.minutes > 0 ? "Resume" : "Play mission"}
          </button>
          <div className="seg" role="group" aria-label="Playback speed">
            {SPEEDS.map((s) => (
              <button key={s} className={clock.speed === s ? "on" : ""} onClick={() => clock.setSpeed(s)}>
                {s} h/s
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Radiation seed">
            {seeds.map((k) => (
              <button key={k} className={seed === k ? "on" : ""} onClick={() => setSeed(k)}>
                Seed {k}
              </button>
            ))}
          </div>
          <span className="hint">Space to play · drag the recorder to scrub</span>
        </div>

        <div className="replay-top">
          <div className="map-frame">
            <GroundMap campaign={campaign} minutes={clock.minutes} theme={theme} inSaa={run.telemetry[step].in_saa} />
          </div>
          <Telemetry campaign={campaign} run={run} minutes={clock.minutes} step={step} />
        </div>

        <div className="replay-block">
          <div className="block-head">
            <h3>Model memory</h3>
            <div className="legend-inline">
              <span><i className="lg lg-crit" /> exponent / sign hit (destructive)</span>
              <span><i className="lg lg-warn" /> moderate</span>
              <span><i className="lg lg-minor" /> negligible</span>
              <span><i className="lg lg-good" /> restored this hour</span>
            </div>
          </div>
          <MemoryMap campaign={campaign} run={run} step={step} />
        </div>

        <div className="replay-block">
          <div className="block-head">
            <h3>Flight recorder</h3>
            <div className="legend-inline">
              <span><i className="lg lg-bar" /> bit flips per hour</span>
              <span><i className="lg lg-crit lg-round" /> exponent-MSB hit</span>
              <span><i className="lg lg-saa" /> SAA pass</span>
              <span><i className="lg lg-accent" /> Aegis alarm</span>
            </div>
          </div>
          <Timeline run={run} stepMinutes={step_minutes} minutes={clock.minutes} onSeek={clock.seek} />
        </div>

        <div className="replay-block">
          <div className="block-head">
            <h3>Accuracy on held-out EuroSAT chips</h3>
            <div className="seg seg-sm" role="group" aria-label="Chart seeds">
              <button className={chartSeed === "follow" ? "on" : ""} onClick={() => setChartSeed("follow")}>
                Seed {seed}
              </button>
              <button className={chartSeed === "mean" ? "on" : ""} onClick={() => setChartSeed("mean")}>
                Mean of {seeds.length}
              </button>
            </div>
          </div>
          <div className="chips" role="group" aria-label="Show or hide strategies">
            {campaign.strategies.map((s) => (
              <button
                key={s.attr}
                className={`chip ${hidden.has(s.attr) ? "chip-off" : ""}`}
                onClick={() => toggle(s.attr)}
                aria-pressed={!hidden.has(s.attr)}
              >
                <span className="swatch" style={{ background: `var(--s-${s.attr})` }} />
                {s.label}
              </button>
            ))}
          </div>
          <AccuracyChart
            campaign={campaign}
            seed={chartSeed === "mean" ? "mean" : seed}
            minutes={clock.minutes}
            hidden={hidden}
            onSeek={clock.seek}
          />
        </div>
      </div>
    </section>
  );
}
