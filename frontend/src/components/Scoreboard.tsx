import { useMemo, useState } from "react";
import type { Campaign, StatRow, StrategyAttr } from "../lib/types";
import { mb, pct, pts } from "../lib/format";
import SectionHead from "./SectionHead";

const ATTR_OF_KEY: Record<string, StrategyAttr> = {
  aegis_v2: "acc_aegis",
  aegis_audit: "acc_aegis_audit",
  ranger_aegis: "acc_ranger_aegis",
  periodic_scrubbing: "acc_periodic_scrub",
  ground_uplink: "acc_uplink",
  ranger_clipping: "acc_ranger",
  blink_recentering: "acc_blink",
  unmitigated: "acc_unmitigated",
};

type SortKey = "mean" | "mb_per_day" | "energy_mj_day";

function CiBar({ row, color }: { row: StatRow; color: string }) {
  const W = 220;
  const sx = (v: number) => 4 + Math.max(0, Math.min(1, v)) * (W - 8);
  return (
    <svg width={W} height={26} className="ci-bar" aria-hidden="true">
      <line x1={4} x2={W - 4} y1={13} y2={13} className="ci-track" />
      {[0.25, 0.5, 0.75].map((t) => (
        <line key={t} x1={sx(t)} x2={sx(t)} y1={9} y2={17} className="ci-tick" />
      ))}
      {row.ci > 0 && (
        <line x1={sx(row.mean - row.ci)} x2={sx(row.mean + row.ci)} y1={13} y2={13} stroke={color} strokeWidth={3} strokeLinecap="round" opacity={0.45} />
      )}
      {row.per_seed.map((v, i) => (
        <circle key={i} cx={sx(v)} cy={13} r={2.4} fill="var(--bg)" stroke={color} strokeWidth={1.2} />
      ))}
      <circle cx={sx(row.mean)} cy={13} r={5} fill={color} className="ring" />
    </svg>
  );
}

export default function Scoreboard({ campaign }: { campaign: Campaign }) {
  const [sort, setSort] = useState<SortKey>("mean");
  const [showSweep, setShowSweep] = useState(false);
  const rows = useMemo(() => {
    const main = campaign.stats.rows.filter((r) => !r.key.startsWith("scrub_") || showSweep);
    const val = (r: StatRow) => (sort === "mean" ? r.mean : sort === "mb_per_day" ? -r.mb_per_day : -(r.energy_mj_day ?? Infinity));
    return [...main].sort((a, b) => val(b) - val(a));
  }, [campaign, sort, showSweep]);

  const colorOf = (r: StatRow) => (ATTR_OF_KEY[r.key] ? `var(--s-${ATTR_OF_KEY[r.key]})` : "var(--s-acc_periodic_scrub)");
  const head = (k: SortKey, label: string) => (
    <th>
      <button className={`th-sort ${sort === k ? "on" : ""}`} onClick={() => setSort(k)} aria-pressed={sort === k}>
        {label}
        {sort === k && <span aria-hidden="true"> ↓</span>}
      </button>
    </th>
  );

  return (
    <section className="section" id="scoreboard">
      <SectionHead
        index="04"
        kicker="Scoreboard"
        title={<>Every strategy, with its <em>uncertainty</em>.</>}
        lede={`Dot = mean over ${campaign.stats.seeds} radiation seeds, bar = 95% Student-t interval, hollow dots = individual seeds. The difference column is paired: every strategy saw the identical fault stream within a seed.`}
      />
      <div className="panel table-panel">
        <div className="table-tools">
          <label className="switch">
            <input type="checkbox" checked={showSweep} onChange={(e) => setShowSweep(e.target.checked)} />
            <span className="switch-track" aria-hidden="true" />
            Include the scrubbing-interval sweep
          </label>
        </div>
        <div className="table-scroll">
          <table className="score">
            <thead>
              <tr>
                <th>Strategy</th>
                {head("mean", "Mean accuracy")}
                <th className="hide-sm">Spread</th>
                <th>End of mission</th>
                {head("mb_per_day", "MB / day")}
                {head("energy_mj_day", "Energy mJ / day")}
                <th>Aegis-v2 minus this</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} className={r.key === "aegis_v2" ? "row-hero" : ""}>
                  <td>
                    <span className="cell-name">
                      <span className="swatch" style={{ background: colorOf(r) }} />
                      {r.label}
                    </span>
                  </td>
                  <td className="mono num">
                    {pct(r.mean)}
                    <span className="ci-text"> ±{(100 * r.ci).toFixed(1)}</span>
                  </td>
                  <td className="hide-sm">
                    <CiBar row={r} color={colorOf(r)} />
                  </td>
                  <td className="mono num">{r.final === null ? "n/a" : pct(r.final)}</td>
                  <td className="mono num">{r.mb_per_day > 0 ? mb(r.mb_per_day) : "0"}</td>
                  <td className="mono num">{r.energy_mj_day === null ? <span className="dim">not charged</span> : r.energy_mj_day.toFixed(0)}</td>
                  <td className="mono num">
                    {r.delta_vs_aegis ? (
                      <>
                        {/* Colored only when the whole 95% interval excludes zero */}
                        <span
                          className={
                            r.delta_vs_aegis.mean - r.delta_vs_aegis.ci > 0
                              ? "pos"
                              : r.delta_vs_aegis.mean + r.delta_vs_aegis.ci < 0
                                ? "neg"
                                : ""
                          }
                        >
                          {pts(r.delta_vs_aegis.mean)} pts
                        </span>
                        <span className="ci-text">
                          {" "}±{(100 * r.delta_vs_aegis.ci).toFixed(1)} · {r.delta_vs_aegis.wins}/{campaign.stats.seeds} seeds
                        </span>
                      </>
                    ) : (
                      <span className="dim">{r.key === "aegis_v2" ? "reference" : ""}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="footnote">
          Energy counts golden-store reads, DRAM writes, checksum scans and the monitor's own activation checks
          {campaign.stats.energy_hook_compute_added ? " (the last added analytically for this campaign)" : ""}. Ranger and BLINK
          compute on every inference is not charged. With 3 seeds the intervals are wide; differences whose interval
          crosses zero are not established.
        </p>
      </div>
    </section>
  );
}
