import { useRef } from "react";
import { motion } from "motion/react";
import { scaleLinear } from "d3-scale";
import { line, curveMonotoneX } from "d3-shape";
import { useElementWidth } from "../../lib/useMissionClock";
import { STRATEGY_COLOR } from "../constants";
import type { MissionState, StrategyKey } from "../types";

const H = 230;
const M = { t: 14, r: 16, b: 28, l: 40 };

/** Accuracy of every strategy over mission time, with SAA steps, scrubs and Aegis restores marked. */
export default function MissionChart({ mission }: { mission: MissionState }) {
  const box = useRef<HTMLDivElement>(null);
  const W = Math.max(320, useElementWidth(box));
  const total = mission.config.total_steps * mission.config.step_minutes;
  const x = scaleLinear().domain([0, total]).range([M.l, W - M.r]);
  const y = scaleLinear().domain([0, 1]).range([H - M.b, M.t]);
  const pts = [{ t: 0, s: null as null | MissionState["steps"][number] }, ...mission.steps.map((s) => ({ t: s.t_min, s }))];
  const keys = mission.strategies.map((s) => s.key);
  const accAt = (k: StrategyKey, p: (typeof pts)[number]) => (p.s ? p.s.strategies[k].acc : mission.clean_acc);
  const hours = total / 60;
  const tickEvery = hours <= 12 ? 2 : hours <= 24 ? 4 : 12;
  const ticks = Array.from({ length: Math.floor(hours / tickEvery) + 1 }, (_, i) => i * tickEvery * 60);
  const now = mission.steps.length ? mission.steps[mission.steps.length - 1].t_min : 0;

  return (
    <div className="m-chart" ref={box}>
      <svg width={W} height={H} role="img" aria-label="Accuracy over mission time per strategy">
        {mission.steps.map((s) =>
          s.in_saa ? (
            <rect
              key={`saa-${s.step}`}
              x={x(s.t_min - mission.config.step_minutes)}
              width={x(s.t_min) - x(s.t_min - mission.config.step_minutes)}
              y={M.t}
              height={H - M.b - M.t}
              className="m-saa"
            />
          ) : null,
        )}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="m-grid" />
            <text x={M.l - 8} y={y(v)} className="m-axis" textAnchor="end" dominantBaseline="middle">
              {v * 100}%
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - 8} className="m-axis" textAnchor="middle">
            {t === 0 ? "T+0" : `${t / 60}h`}
          </text>
        ))}
        <line x1={M.l} x2={W - M.r} y1={y(mission.clean_acc)} y2={y(mission.clean_acc)} className="m-clean" />
        {keys.map((k) => {
          const d = line<(typeof pts)[number]>()
            .x((p) => x(p.t))
            .y((p) => y(accAt(k, p)))
            .curve(curveMonotoneX)(pts);
          const lastP = pts[pts.length - 1];
          return (
            <g key={k}>
              <path d={d ?? ""} className={`m-line ${k === "aegis" || k === "audit" ? "strong" : ""}`} style={{ stroke: STRATEGY_COLOR[k] }} />
              {mission.running && (
                <circle cx={x(lastP.t)} cy={y(accAt(k, lastP))} r={3.2} className="m-head" style={{ fill: STRATEGY_COLOR[k] }} />
              )}
            </g>
          );
        })}
        {mission.steps.map((s) => (
          <g key={`ev-${s.step}`}>
            {s.strategies.scrub.scrubbed && <line x1={x(s.t_min)} x2={x(s.t_min)} y1={M.t} y2={H - M.b} className="m-scrub" />}
            {s.strategies.aegis.restored.length > 0 && (
              <motion.circle
                cx={x(s.t_min)}
                cy={y(s.strategies.aegis.acc)}
                r={4}
                className="m-restore"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 18 }}
              />
            )}
          </g>
        ))}
        {mission.running && <line x1={x(now)} x2={x(now)} y1={M.t} y2={H - M.b} className="m-now" />}
      </svg>
      <div className="m-legend">
        {mission.strategies.map((s) => (
          <span key={s.key}>
            <i style={{ background: STRATEGY_COLOR[s.key] }} />
            {s.label}
          </span>
        ))}
        <span>
          <i className="lg-saa-box" />
          SAA step
        </span>
        <span>
          <i className="lg-scrub-line" />
          scrub
        </span>
        <span>
          <i className="lg-restore-dot" />
          Aegis restore
        </span>
      </div>
    </div>
  );
}
