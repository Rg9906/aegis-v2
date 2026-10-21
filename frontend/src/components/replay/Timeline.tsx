import { useRef, type PointerEvent } from "react";
import { scaleLinear } from "d3-scale";
import type { SeedRun } from "../../lib/types";
import { useElementWidth } from "../../lib/useMissionClock";

const H = 64;

interface Props {
  run: SeedRun;
  stepMinutes: number;
  minutes: number;
  onSeek: (minutes: number) => void;
}

/** Flight recorder: flips per hour, exponent-MSB hits, SAA passes, Aegis alarms. Drag to scrub. */
export default function Timeline({ run, stepMinutes, minutes, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const width = useElementWidth(wrap);
  const n = run.telemetry.length;
  const total = n * stepMinutes;
  const x = scaleLinear().domain([0, total]).range([0, Math.max(width, 1)]);
  const maxFlips = Math.max(...run.telemetry.map((t) => t.bit_flips_step), 1);
  const bw = Math.max(1, x(stepMinutes) - x(0) - 1);

  const seekFrom = (e: PointerEvent<HTMLDivElement>) => {
    const box = wrap.current!.getBoundingClientRect();
    onSeek(x.invert(e.clientX - box.left));
  };

  return (
    <div
      className="timeline"
      ref={wrap}
      role="slider"
      tabIndex={0}
      aria-label="Mission time"
      aria-valuemin={0}
      aria-valuemax={Math.round(total)}
      aria-valuenow={Math.round(minutes)}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFrom(e);
      }}
      onPointerMove={(e) => {
        if (e.buttons) seekFrom(e);
      }}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") onSeek(minutes + stepMinutes);
        if (e.key === "ArrowLeft") onSeek(minutes - stepMinutes);
      }}
    >
      {width > 0 && (
        <svg width={width} height={H} aria-hidden="true">
          {run.telemetry.map((t, i) => {
            const x0 = x(i * stepMinutes);
            const h = (t.bit_flips_step / maxFlips) * (H - 26);
            return (
              <g key={i}>
                {t.saa_minutes_in_step > 0 && (
                  <rect x={x0} y={0} width={bw + 1} height={H - 10} className="saa-band"
                    style={{ opacity: 0.3 + 0.7 * Math.min(1, (t.saa_minutes_in_step / stepMinutes) * 3) }} />
                )}
                <rect x={x0} y={H - 12 - h} width={bw} height={h} rx={1}
                  className={x0 <= x(minutes) ? "flip-bar flip-bar-past" : "flip-bar"} />
                {t.exponent_msb_flips_step > 0 && (
                  <circle cx={x0 + bw / 2} cy={H - 16 - h} r={2.6} className="msb-dot" />
                )}
                {t.aegis_alarm && <rect x={x0} y={H - 6} width={bw} height={3} rx={1} className="alarm-mark" />}
              </g>
            );
          })}
          <line x1={x(minutes)} x2={x(minutes)} y1={0} y2={H} className="playhead" />
        </svg>
      )}
    </div>
  );
}
