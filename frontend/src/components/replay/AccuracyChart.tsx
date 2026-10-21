import { useMemo, useRef, useState, type MouseEvent } from "react";
import { scaleLinear } from "d3-scale";
import { area, curveStepAfter, line } from "d3-shape";
import type { Campaign, StrategyAttr } from "../../lib/types";
import { accuracySeries, seedKeys } from "../../lib/data";
import { missionClock, pct } from "../../lib/format";
import { useElementWidth } from "../../lib/useMissionClock";
import Tooltip, { type TipState } from "../Tooltip";

const H = 300;
const M = { top: 14, right: 16, bottom: 30, left: 44 };

interface Props {
  campaign: Campaign;
  seed: string | "mean";
  minutes: number;
  hidden: Set<StrategyAttr>;
  onSeek: (minutes: number) => void;
}

export default function AccuracyChart({ campaign, seed, minutes, hidden, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const width = useElementWidth(wrap);
  const [hover, setHover] = useState<number | null>(null);
  const [tip, setTip] = useState<TipState | null>(null);
  const { steps, step_minutes } = campaign.setup;
  const total = steps * step_minutes;

  const x = useMemo(() => scaleLinear().domain([0, total]).range([M.left, Math.max(width, 300) - M.right]), [total, width]);
  const y = useMemo(() => scaleLinear().domain([0, 1]).range([H - M.bottom, M.top]), []);
  // Step k's accuracy is measured at the END of the hour, so it holds over [k*dt, (k+1)*dt)
  const tAt = (i: number) => i * step_minutes;

  const series = useMemo(
    () =>
      campaign.strategies.map((s) => ({
        s,
        data: accuracySeries(campaign, s.attr, seed),
      })),
    [campaign, seed],
  );
  const ref = seed === "mean" ? campaign.seeds[seedKeys(campaign)[0]] : campaign.seeds[seed];
  const saaSteps = ref.telemetry.map((t, i) => ({ i, w: t.saa_minutes_in_step / step_minutes }));

  const lineGen = line<{ i: number; mean: number }>()
    .x((d) => x(tAt(d.i)))
    .y((d) => y(d.mean))
    .curve(curveStepAfter);
  const bandGen = area<{ i: number; lo: number; hi: number }>()
    .x((d) => x(tAt(d.i)))
    .y0((d) => y(d.lo))
    .y1((d) => y(d.hi))
    .curve(curveStepAfter);
  const extend = <T extends { i: number }>(d: T[]) => [...d, { ...d[d.length - 1], i: d.length }];

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const box = wrap.current!.getBoundingClientRect();
    const mx = e.clientX - box.left;
    const m = Math.max(0, Math.min(total - 1e-6, x.invert(mx)));
    const i = Math.min(steps - 1, Math.floor(m / step_minutes));
    setHover(i);
    const vals = series
      .filter(({ s }) => !hidden.has(s.attr))
      .map(({ s, data }) => ({ s, v: data[i].mean }))
      .sort((a, b) => b.v - a.v);
    setTip({
      x: mx,
      y: e.clientY - box.top - 40,
      content: (
        <>
          <div className="tip-title">
            {missionClock((i + 1) * step_minutes)}
            {ref.telemetry[i].in_saa && <span className="badge badge-saa">SAA</span>}
          </div>
          {vals.map(({ s, v }) => (
            <div className="tip-row tip-series" key={s.attr}>
              <span className="swatch" style={{ background: `var(--s-${s.attr})` }} />
              <span className="tip-name">{s.label}</span>
              <span className="mono">{pct(v)}</span>
            </div>
          ))}
        </>
      ),
    });
  };

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const dayTicks = Array.from({ length: Math.floor(campaign.setup.days) + 1 }, (_, d) => d * 1440);

  return (
    <div className="acc-chart" ref={wrap} onMouseLeave={() => { setHover(null); setTip(null); }}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Accuracy of every strategy over the mission">
          {saaSteps.map(({ i, w }) =>
            w > 0 ? (
              <rect key={i} x={x(tAt(i))} y={M.top} width={x(tAt(i + 1)) - x(tAt(i))} height={H - M.top - M.bottom}
                className="saa-band" style={{ opacity: 0.25 + 0.75 * Math.min(1, w * 3) }} />
            ) : null,
          )}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} className="gridline" />
              <text x={M.left - 8} y={y(t) + 4} textAnchor="end" className="axis-label">
                {t * 100}%
              </text>
            </g>
          ))}
          {dayTicks.map((m) => (
            <text key={m} x={x(m)} y={H - 8} textAnchor="middle" className="axis-label">
              Day {m / 1440}
            </text>
          ))}
          <line x1={M.left} x2={width - M.right} y1={y(campaign.setup.clean_accuracy)} y2={y(campaign.setup.clean_accuracy)}
            className="ref-line" />
          <text x={width - M.right} y={y(campaign.setup.clean_accuracy) - 6} textAnchor="end" className="axis-label">
            clean model {pct(campaign.setup.clean_accuracy)}
          </text>

          {seed === "mean" &&
            series.map(({ s, data }) =>
              hidden.has(s.attr) ? null : (
                <path key={s.attr} d={bandGen(extend(data)) ?? ""} fill={`var(--s-${s.attr})`} opacity={0.07} />
              ),
            )}
          {series.map(({ s, data }) =>
            hidden.has(s.attr) ? null : (
              <path
                key={s.attr}
                d={lineGen(extend(data)) ?? ""}
                fill="none"
                stroke={`var(--s-${s.attr})`}
                strokeWidth={s.attr === "acc_aegis" ? 2.6 : 1.8}
                strokeLinejoin="round"
                opacity={hover === null ? 1 : 0.9}
              />
            ),
          )}

          <line x1={x(minutes)} x2={x(minutes)} y1={M.top - 6} y2={H - M.bottom} className="playhead" />
          <circle cx={x(minutes)} cy={M.top - 6} r={3.5} className="playhead-knob" />
          {hover !== null && (
            <line x1={x(tAt(hover + 0.5))} x2={x(tAt(hover + 0.5))} y1={M.top} y2={H - M.bottom} className="crosshair" />
          )}
          <rect
            x={M.left}
            y={M.top}
            width={Math.max(0, width - M.left - M.right)}
            height={H - M.top - M.bottom}
            fill="transparent"
            onMouseMove={onMove}
            onClick={(e) => {
              const box = wrap.current!.getBoundingClientRect();
              onSeek(x.invert(e.clientX - box.left));
            }}
            style={{ cursor: "crosshair" }}
          />
        </svg>
      )}
      <Tooltip tip={tip} width={width} />
    </div>
  );
}
