import { useMemo, useRef, useState } from "react";
import type { Campaign, SeedRun } from "../../lib/types";
import { MEMORY_STRATEGIES, memoryState, type BlockState, type Severity } from "../../lib/data";
import { blockShort, missionClock, pct } from "../../lib/format";
import { useElementWidth } from "../../lib/useMissionClock";
import Tooltip, { type TipState } from "../Tooltip";

const ROW_H = 40;
const ROW_GAP = 18;
const LABEL_W = 168;
const TICK_W = 2;
const TICK_GAP = 2;

const sevColor: Record<Severity, string> = {
  critical: "var(--crit)",
  moderate: "var(--warn)",
  minor: "var(--ink-4)",
};

function worst(s: BlockState): Severity | null {
  if (s.hits.some((x) => x.sev === "critical")) return "critical";
  if (s.hits.some((x) => x.sev === "moderate")) return "moderate";
  return s.hits.length ? "minor" : null;
}

interface Props {
  campaign: Campaign;
  run: SeedRun;
  step: number;
}

/**
 * Model memory, block by block, for strategies whose repairs are recorded per step.
 * Block width ~ sqrt(size); ticks = bit flips since the block was last clean, in arrival order.
 */
export default function MemoryMap({ campaign, run, step }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const width = useElementWidth(wrap);
  const [tip, setTip] = useState<TipState | null>(null);
  const blockIds = campaign.blocks.map((b) => b.id);

  // Narrow screens: labels sit above each row and blocks take the full width
  const narrow = width > 0 && width < 640;
  const labelW = narrow ? 0 : LABEL_W;
  const labelH = narrow ? 30 : 0;

  const layout = useMemo(() => {
    const avail = Math.max(width - labelW, 120);
    const gaps = 2 * (campaign.blocks.length - 1);
    const weights = campaign.blocks.map((b) => Math.sqrt(b.bytes));
    const total = weights.reduce((a, b) => a + b, 0);
    const minW = narrow ? 12 : 22;
    const raw = weights.map((w) => ((avail - gaps) * w) / total);
    // Lift tiny blocks to a minimum width, shrinking the rest proportionally
    const deficit = raw.reduce((a, w) => a + Math.max(0, minW - w), 0);
    const bigTotal = raw.filter((w) => w >= minW).reduce((a, b) => a + b, 0);
    let x = labelW;
    return campaign.blocks.map((b, i) => {
      const w = raw[i] < minW ? minW : raw[i] - (deficit * raw[i]) / bigTotal;
      const cell = { id: b.id, bytes: b.bytes, x, w };
      x += w + 2;
      return cell;
    });
  }, [campaign.blocks, width, labelW, narrow]);

  const rows = MEMORY_STRATEGIES.filter((s) => campaign.strategies.some((c) => c.attr === s.attr)).map((s) => ({
    ...s,
    state: memoryState(run.telemetry, run.faults, blockIds, s.attr, step),
    acc: run.telemetry[step]?.acc[s.attr],
  }));
  const height = 22 + rows.length * (ROW_H + ROW_GAP + labelH);

  const showTip = (e: React.MouseEvent, id: string, label: string, s: BlockState, bytes: number) => {
    const box = wrap.current!.getBoundingClientRect();
    const sorted = [...s.hits].sort((a, b) => ["critical", "moderate", "minor"].indexOf(a.sev) - ["critical", "moderate", "minor"].indexOf(b.sev));
    setTip({
      x: e.clientX - box.left,
      y: e.clientY - box.top,
      content: (
        <>
          <div className="tip-title">
            {label} · {blockShort(id)} <span className="tip-dim">{(bytes / 1e6).toFixed(2)} MB</span>
          </div>
          {s.hits.length === 0 ? (
            <div className="tip-row">Bit-exact with the golden copy{s.restoredAt !== null ? ` (restored ${missionClock((s.restoredAt + 1) * campaign.setup.step_minutes)})` : ""}</div>
          ) : (
            <>
              <div className="tip-row">{s.hits.length} flip{s.hits.length > 1 ? "s" : ""} since last clean</div>
              {sorted.slice(0, 4).map(({ h, step: st, sev }, k) => (
                <div className="tip-row mono" key={k}>
                  <span className="sev-dot" style={{ background: sevColor[sev] }} />
                  {h.p.split(".").slice(-2).join(".")} bit {h.bit}: {h.from} → {String(h.to)}{" "}
                  <span className="tip-dim">{missionClock((st + 1) * campaign.setup.step_minutes)}</span>
                </div>
              ))}
              {s.hits.length > 4 && <div className="tip-row tip-dim">+{s.hits.length - 4} more</div>}
            </>
          )}
        </>
      ),
    });
  };

  return (
    <div className="memmap" ref={wrap} onMouseLeave={() => setTip(null)}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Model memory blocks and the bit flips in each">
          {layout.map((c) => (
            <text key={c.id} x={c.x + c.w / 2} y={12} className="axis-label" textAnchor="middle">
              {c.w > 30 ? blockShort(c.id) : c.w > 16 ? blockShort(c.id).replace("·", "") : ""}
            </text>
          ))}
          {rows.map((r, ri) => {
            const top = 22 + ri * (ROW_H + ROW_GAP + labelH);
            const y = top + labelH;
            return (
              <g key={r.attr}>
                {narrow ? (
                  <>
                    <rect x={0} y={top + 4} width={3} height={16} rx={1.5} fill={`var(--s-${r.attr})`} />
                    <text x={10} y={top + 17} className="row-label">
                      {r.label}
                      <tspan className="row-sub" dx={8}>{r.acc !== undefined ? pct(r.acc) : ""}</tspan>
                    </text>
                  </>
                ) : (
                  <>
                    <rect x={0} y={y + 6} width={3} height={ROW_H - 12} rx={1.5} fill={`var(--s-${r.attr})`} />
                    <text x={12} y={y + 17} className="row-label">
                      {r.label}
                    </text>
                    <text x={12} y={y + 33} className="row-sub">
                      {r.acc !== undefined ? `accuracy ${pct(r.acc)}` : ""}
                    </text>
                  </>
                )}
                {layout.map((c) => {
                  const s = r.state[c.id];
                  const sev = worst(s);
                  const justRestored = s.restoredAt === step;
                  const maxTicks = Math.max(0, Math.floor((c.w - 6) / (TICK_W + TICK_GAP)));
                  return (
                    <g
                      key={c.id}
                      onMouseMove={(e) => showTip(e, c.id, r.label, s, c.bytes)}
                      style={{ cursor: "default" }}
                    >
                      <rect
                        x={c.x}
                        y={y}
                        width={c.w}
                        height={ROW_H}
                        rx={4}
                        className={`cell ${sev ? `cell-${sev}` : "cell-clean"} ${justRestored ? "cell-restored" : ""}`}
                      />
                      {s.hits.slice(0, maxTicks).map((hit, k) => (
                        <rect
                          key={k}
                          x={c.x + 3 + k * (TICK_W + TICK_GAP)}
                          y={y + (hit.sev === "critical" ? 5 : hit.sev === "moderate" ? 11 : 17)}
                          width={TICK_W}
                          height={hit.sev === "critical" ? ROW_H - 10 : hit.sev === "moderate" ? ROW_H - 22 : ROW_H - 34}
                          rx={1}
                          fill={sevColor[hit.sev]}
                          className={hit.step === step ? "tick-new" : undefined}
                        />
                      ))}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      )}
      <Tooltip tip={tip} width={width} />
    </div>
  );
}
