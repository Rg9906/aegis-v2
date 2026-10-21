import type { Campaign, SeedRun } from "../../lib/types";
import { positionAt } from "../../lib/geo";
import { blockShort, latLon, missionClock } from "../../lib/format";

interface Props {
  campaign: Campaign;
  run: SeedRun;
  minutes: number;
  step: number;
}

/** Live readout for the replay: clock, position, environment, Aegis status. */
export default function Telemetry({ campaign, run, minutes, step }: Props) {
  const t = run.telemetry[step];
  const pos = campaign.track ? positionAt(campaign.track, minutes) : null;
  const cumulative = run.telemetry.slice(0, step + 1).reduce((a, s) => a + s.bit_flips_step, 0);
  const cumulativeMsb = run.telemetry.slice(0, step + 1).reduce((a, s) => a + s.exponent_msb_flips_step, 0);
  const alarm = t.aegis_alarm;

  return (
    <div className="telemetry">
      <div className="tel-clock mono">{missionClock(minutes)}</div>
      <div className="tel-grid">
        <div className="tel-item">
          <span className="tel-k">Position</span>
          <span className="tel-v mono">{pos ? latLon(pos[1], pos[0]) : "synthetic orbit"}</span>
        </div>
        <div className="tel-item">
          <span className="tel-k">Altitude</span>
          <span className="tel-v mono">{campaign.track ? `${campaign.track.alt_km.toFixed(0)} km` : "n/a"}</span>
        </div>
        <div className="tel-item">
          <span className="tel-k">Environment</span>
          <span className={`tel-v ${t.in_saa ? "tel-saa" : ""}`}>
            {t.in_saa ? `In SAA · ${t.saa_minutes_in_step.toFixed(0)} min this hour` : "Background flux"}
          </span>
        </div>
        <div className="tel-item">
          <span className="tel-k">Bit flips this hour</span>
          <span className="tel-v mono">
            {t.bit_flips_step}
            {t.exponent_msb_flips_step > 0 && <span className="crit-text"> · {t.exponent_msb_flips_step} exponent MSB</span>}
          </span>
        </div>
        <div className="tel-item">
          <span className="tel-k">Since launch</span>
          <span className="tel-v mono">
            {cumulative} flips · {cumulativeMsb} MSB
          </span>
        </div>
      </div>
      <div className={`tel-status ${alarm ? "tel-status-alarm" : ""}`}>
        <span className="status-icon" aria-hidden="true">{alarm ? "◆" : "●"}</span>
        <div>
          <div className="tel-status-title">{alarm ? "Aegis-v2: damage confirmed, repaired" : "Aegis-v2: nominal"}</div>
          <div className="tel-status-sub">
            {alarm
              ? t.aegis_restored_blocks.length
                ? `Restored ${t.aegis_restored_blocks.map(blockShort).join(", ")} from the golden copy`
                : "Alarm raised; no block failed its checksum"
              : "Activations and batch-norm statistics within calibrated bounds"}
          </div>
        </div>
      </div>
    </div>
  );
}
