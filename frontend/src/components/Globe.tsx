import { useEffect, useMemo, useRef } from "react";
import { geoDistance, geoGraticule10, geoOrthographic, geoPath } from "d3-geo";
import type { Campaign } from "../lib/types";
import { landDots, positionAt, saaWeight } from "../lib/geo";
import { cssVar, usePrefersReducedMotion } from "../lib/useMissionClock";

interface Props {
  campaign: Campaign;
  theme: string;
}

const ORBIT_SECONDS = 14; // one real-time revolution of the satellite in the hero

/** Rotating orthographic globe: halftone land, SAA glow from IGRF |B|, live orbit arc. */
export default function Globe({ campaign, theme }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();
  const track = campaign.track;

  const saaPoints = useMemo(() => {
    const f = campaign.field;
    if (!f) return [];
    const pts: [number, number, number][] = [];
    f.lats.forEach((lat, i) =>
      f.lons.forEach((lon, j) => {
        const w = saaWeight(f.b_nt[i][j], f.edge_nt, f.core_nt);
        if (w > 0) pts.push([lon, lat, w]);
      }),
    );
    return pts;
  }, [campaign.field]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dots = landDots();
    const colors = {
      land: cssVar("--land"),
      landFar: cssVar("--land-far"),
      grat: cssVar("--graticule"),
      line: cssVar("--line-strong"),
      accent: cssVar("--accent"),
      ink: cssVar("--ink"),
      saa: cssVar("--saa"),
    };
    let raf = 0;
    let start: number | null = null;

    const draw = (now: number) => {
      if (start === null) start = now;
      const t = reduced ? 0 : (now - start) / 1000;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = canvas.clientWidth;
      if (canvas.width !== Math.round(size * dpr)) {
        canvas.width = Math.round(size * dpr);
        canvas.height = Math.round(size * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);

      const r = size / 2 - 6;
      const period = track?.period_min ?? 100;
      const satMinutes = ((t / ORBIT_SECONDS) * period) % (track ? track.minutes[track.minutes.length - 1] : 1);
      const sat = track ? positionAt(track, satMinutes) : ([0, 0] as [number, number]);
      // Slow steady spin, tilted to look slightly south so the SAA reads well; starts over the Atlantic
      const rotate: [number, number, number] = [35 - t * 7, 14, 0];
      const proj = geoOrthographic().scale(r).translate([size / 2, size / 2]).rotate(rotate).clipAngle(90);
      const path = geoPath(proj, ctx);
      const center: [number, number] = [-rotate[0], -rotate[1]];

      // Atmosphere rim
      const glow = ctx.createRadialGradient(size / 2, size / 2, r * 0.9, size / 2, size / 2, r * 1.06);
      glow.addColorStop(0, "rgba(0,0,0,0)");
      glow.addColorStop(1, colors.accent + "22");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(size / 2, size / 2, r * 1.06, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      path({ type: "Sphere" });
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.beginPath();
      path(geoGraticule10());
      ctx.strokeStyle = colors.grat;
      ctx.lineWidth = 0.6;
      ctx.stroke();

      // SAA as halftone: dot size and opacity follow the radiation weight
      for (const [lon, lat, w] of saaPoints) {
        const dist = geoDistance([lon, lat], center);
        if (dist > Math.PI / 2) continue;
        const p = proj([lon, lat]);
        if (!p) continue;
        const limb = Math.cos(dist);
        ctx.fillStyle = `rgba(${colors.saa}, ${(0.25 + 0.6 * w) * (0.4 + 0.6 * limb)})`;
        ctx.beginPath();
        ctx.arc(p[0], p[1], (0.9 + 2.4 * w) * (0.55 + 0.45 * limb) * (size / 600), 0, Math.PI * 2);
        ctx.fill();
      }

      // Halftone land, dimmer toward the limb
      for (const d of dots) {
        const dist = geoDistance(d, center);
        if (dist > Math.PI / 2) continue;
        const p = proj(d)!;
        ctx.fillStyle = dist > 1.2 ? colors.landFar : colors.land;
        ctx.fillRect(p[0] - 0.8, p[1] - 0.8, 1.6, 1.6);
      }

      if (track) {
        // Orbit arc: half a revolution behind (fading) and a faint lead ahead
        const trail = 0.55 * period;
        const n = 90;
        for (let k = 0; k < n; k++) {
          const m0 = satMinutes - trail + (k * trail) / n;
          const m1 = m0 + trail / n;
          if (m0 < 0) continue;
          const a = positionAt(track, m0);
          const b = positionAt(track, m1);
          if (geoDistance(a, center) > Math.PI / 2 || geoDistance(b, center) > Math.PI / 2) continue;
          const pa = proj(a)!;
          const pb = proj(b)!;
          ctx.strokeStyle = colors.accent;
          ctx.globalAlpha = (k / n) ** 1.6;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.setLineDash([2, 5]);
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let started = false;
        for (let k = 0; k <= 40; k++) {
          const q = positionAt(track, satMinutes + (k * 0.35 * period) / 40);
          if (geoDistance(q, center) > Math.PI / 2) {
            started = false;
            continue;
          }
          const p = proj(q)!;
          if (started) ctx.lineTo(p[0], p[1]);
          else ctx.moveTo(p[0], p[1]);
          started = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);

        if (geoDistance(sat, center) <= Math.PI / 2) {
          const p = proj(sat)!;
          const pulse = reduced ? 0.5 : (Math.sin(t * 3) + 1) / 2;
          ctx.fillStyle = colors.accent + "33";
          ctx.beginPath();
          ctx.arc(p[0], p[1], 9 + pulse * 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = colors.ink;
          ctx.beginPath();
          ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      if (!reduced) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [saaPoints, track, theme, reduced]);

  return (
    <canvas
      ref={ref}
      className="globe"
      role="img"
      aria-label="Rotating globe showing the Sentinel-2A orbit and the South Atlantic Anomaly radiation region"
    />
  );
}
