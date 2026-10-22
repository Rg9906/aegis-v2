import { useMemo, useRef, useState, type MouseEvent } from "react";
import { scaleLinear, scaleLog } from "d3-scale";
import type { Campaign, ParetoPoint, StrategyAttr } from "../lib/types";
import { mb, pct } from "../lib/format";
import { useElementWidth } from "../lib/useMissionClock";
import Tooltip, { type TipState } from "./Tooltip";
import SectionHead from "./SectionHead";

const H = 440;
const M = { top: 20, right: 28, bottom: 52, left: 56 };
const ZERO_COL = 46; // zero-traffic strategies sit in their own column left of the log axis

const ATTR_OF: Record<string, StrategyAttr> = {
  "Aegis-v2 selective restore": "acc_aegis",
  "Aegis-v2 + CRC audit": "acc_aegis_audit",
  "Ranger + Aegis-v2": "acc_ranger_aegis",
  "Ground uplink reload": "acc_uplink",
  "Ranger clipping": "acc_ranger",
  "BLINK re-centering": "acc_blink",
  Unmitigated: "acc_unmitigated",
};

export default function Pareto({ campaign }: { campaign: Campaign }) {
  const wrap = useRef<HTMLDivElement>(null);
  const width = useElementWidth(wrap);
  const [tip, setTip] = useState<TipState | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const pts = campaign.stats.pareto;
  const scrub = pts.filter((p) => p.family === "scrub").sort((a, b) => a.mb_per_day - b.mb_per_day);
  const others = pts.filter((p) => p.family !== "scrub");
  const verdict = campaign.stats.verdicts.find((v) => v.label === "Aegis-v2 selective restore");

  const maxMb = Math.max(...pts.map((p) => p.mb_per_day), 100);
  const minMb = Math.min(...pts.filter((p) => p.mb_per_day > 0).map((p) => p.mb_per_day), 10);
  const x = useMemo(
    () =>
      scaleLog()
        .domain([Math.pow(10, Math.floor(Math.log10(minMb))), Math.pow(10, Math.ceil(Math.log10(maxMb)))])
        .range([M.left + ZERO_COL + 18, Math.max(width, 360) - M.right]),
    [minMb, maxMb, width],
  );
  const y = scaleLinear().domain([0, 1]).range([H - M.bottom, M.top]);
  const px = (p: ParetoPoint) => (p.mb_per_day > 0 ? x(p.mb_per_day) : M.left + ZERO_COL / 2);
  const zeroJitter = (p: ParetoPoint, i: number) => (p.mb_per_day > 0 ? 0 : (i % 2 ? 7 : -7));

  const show = (e: MouseEvent, p: ParetoPoint) => {
    const box = wrap.current!.getBoundingClientRect();
    setFocus(p.strategy);
    setTip({
      x: e.clientX - box.left,
      y: e.clientY - box.top,
      content: (
        <>
          <div className="tip-title">{p.strategy}</div>
          <div className="tip-row">
            Mean accuracy <span className="mono">{pct(p.mean_accuracy)}</span>
          </div>
          <div className="tip-row">
            Repair traffic <span className="mono">{mb(p.mb_per_day)} MB/day</span>
          </div>
        </>
      ),
    });
  };

  const decades: number[] = [];
  for (let d = x.domain()[0]; d <= x.domain()[1]; d *= 10) decades.push(d);
  const match = verdict?.scrub_mb_to_match ?? null;

  return (
    <section className="section" id="tradeoff">
      <SectionHead
        index="04"
        kicker="The trade-off"
        title={<>Same accuracy, a <em>fraction</em> of the traffic.</>}
        lede="Periodic scrubbing buys accuracy by reloading the whole model on a timer; the only knob is how often. Each orange point is one interval. A strategy above and left of that curve gets more accuracy per megabyte moved."
      />
      <div className="tradeoff-grid">
        <div className="panel chart-panel" ref={wrap} onMouseLeave={() => { setTip(null); setFocus(null); }}>
          {width > 0 && (
            <svg width={width} height={H} role="img" aria-label="Accuracy versus repair traffic for every strategy">
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
                <g key={t}>
                  <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} className="gridline" />
                  <text x={M.left - 10} y={y(t) + 4} textAnchor="end" className="axis-label">{t * 100}%</text>
                </g>
              ))}
              {decades.map((d) => (
                <g key={d}>
                  <line x1={x(d)} x2={x(d)} y1={M.top} y2={H - M.bottom} className="gridline gridline-v" />
                  <text x={x(d)} y={H - M.bottom + 20} textAnchor="middle" className="axis-label">{d >= 1000 ? `${d / 1000}k` : d}</text>
                </g>
              ))}
              <text x={M.left + ZERO_COL / 2} y={H - M.bottom + 20} textAnchor="middle" className="axis-label">0</text>
              <line x1={M.left + ZERO_COL + 6} x2={M.left + ZERO_COL + 6} y1={M.top} y2={H - M.bottom} className="axis-break" />
              <text x={(M.left + width - M.right) / 2} y={H - 10} textAnchor="middle" className="axis-title">
                Repair traffic from the golden copy (MB per day, log scale)
              </text>
              <text transform={`translate(16 ${(H - M.bottom + M.top) / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">
                Time-averaged accuracy
              </text>

              {verdict && match && (
                <g className="match-annot">
                  <line x1={x(verdict.mb_per_day)} x2={x(match)} y1={y(verdict.mean_accuracy)} y2={y(verdict.mean_accuracy)} />
                  <circle cx={x(match)} cy={y(verdict.mean_accuracy)} r={4} />
                  <text x={x(match)} y={y(verdict.mean_accuracy) - 10} textAnchor="end">
                    {width < 640
                      ? `${verdict.traffic_ratio?.toFixed(1)}× the traffic`
                      : `scrubbing needs ${mb(match)} MB/day · ${verdict.traffic_ratio?.toFixed(1)}×`}
                  </text>
                </g>
              )}

              <path
                d={scrub.map((p, i) => `${i ? "L" : "M"}${px(p)},${y(p.mean_accuracy)}`).join("")}
                fill="none"
                stroke="var(--s-acc_periodic_scrub)"
                strokeWidth={2}
              />
              {scrub.map((p) => (
                <g key={p.strategy} onMouseMove={(e) => show(e, p)}>
                  <circle cx={px(p)} cy={y(p.mean_accuracy)} r={12} fill="transparent" />
                  <circle cx={px(p)} cy={y(p.mean_accuracy)} r={5} fill="var(--s-acc_periodic_scrub)" className="ring" />
                  <text x={px(p) + 9} y={y(p.mean_accuracy) + 16} className="pt-label">{p.interval_h} h</text>
                </g>
              ))}
              {others.map((p, i) => {
                const attr = ATTR_OF[p.strategy];
                const isAegis = p.family === "aegis";
                const cx = px(p) + zeroJitter(p, i);
                const dim = focus !== null && focus !== p.strategy;
                return (
                  <g key={p.strategy} onMouseMove={(e) => show(e, p)} opacity={dim ? 0.45 : 1}>
                    <circle cx={cx} cy={y(p.mean_accuracy)} r={14} fill="transparent" />
                    {isAegis && <circle cx={cx} cy={y(p.mean_accuracy)} r={13} fill={`var(--s-${attr})`} opacity={0.14} />}
                    {p.family === "baseline" || p.family === "uplink" ? (
                      <rect x={cx - 5} y={y(p.mean_accuracy) - 5} width={10} height={10} rx={2} fill={`var(--s-${attr})`} className="ring" />
                    ) : (
                      <circle cx={cx} cy={y(p.mean_accuracy)} r={isAegis ? 7 : 5} fill={`var(--s-${attr})`} className="ring" />
                    )}
                  </g>
                );
              })}
            </svg>
          )}
          <Tooltip tip={tip} width={width} />
        </div>

        <aside className="tradeoff-legend">
          <h3>Strategies</h3>
          <ul>
            <li onMouseEnter={() => setFocus("scrub")} onMouseLeave={() => setFocus(null)}>
              <span className="swatch" style={{ background: "var(--s-acc_periodic_scrub)" }} />
              Periodic scrubbing, by interval
            </li>
            {others.map((p) => (
              <li key={p.strategy} onMouseEnter={() => setFocus(p.strategy)} onMouseLeave={() => setFocus(null)}>
                <span
                  className={`swatch ${p.family === "baseline" || p.family === "uplink" ? "swatch-sq" : ""}`}
                  style={{ background: `var(--s-${ATTR_OF[p.strategy]})` }}
                />
                {p.strategy}
                <span className="legend-val mono">{pct(p.mean_accuracy)}</span>
              </li>
            ))}
          </ul>
          {campaign.stats.verdicts.map((v) => (
            <div className="verdict" key={v.label}>
              <div className="verdict-title">{v.label}</div>
              <div className="verdict-body">
                {v.dominated_by.length === 0 ? "Not beaten by any scrubbing interval on both axes." : `Beaten on both axes by ${v.dominated_by.join(", ")}.`}
                {v.traffic_ratio ? ` Scrubbing needs ${v.traffic_ratio.toFixed(1)}× the traffic to match its accuracy.` : ""}
              </div>
            </div>
          ))}
        </aside>
      </div>
    </section>
  );
}
