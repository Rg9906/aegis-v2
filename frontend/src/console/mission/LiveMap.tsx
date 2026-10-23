import { useEffect, useMemo, useRef } from "react";
import { contours } from "d3-contour";
import { landDots, positionAt } from "../../lib/geo";
import { cssVar } from "../../lib/useMissionClock";
import type { MissionState, MissionStep } from "../types";

const LAT_MAX = 84;
const LAT_MIN = -84;
const TRAIL_MIN = 200;

interface Burst {
  lon: number;
  lat: number;
  born: number;
  n: number;
  crit: boolean;
  seeds: number[];
}

/**
 * Live equirectangular map: SAA isobands (IGRF |B|), halftone land, the real Sentinel-2A track,
 * and the satellite gliding toward the next step while the engine computes it. When a step completes,
 * its flips burst where they struck along that step's path (clustered in time; size ~ flips, red if
 * an exponent-MSB hit is in the cluster) and leave a faint scar for the rest of the mission.
 */
export default function LiveMap({ mission, theme }: { mission: MissionState; theme: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const track = mission.track;
  const field = mission.field;
  const stepMin = mission.config.step_minutes;
  const steps = mission.steps;

  const bands = useMemo(() => {
    if (!field) return [];
    const nx = field.lons.length;
    const ny = field.lats.length;
    const step = field.lons[1] - field.lons[0];
    const values = field.lats.flatMap((_, i) => field.lons.map((__, j) => -field.b_nt[i][j]));
    return contours()
      .size([nx, ny])
      .thresholds([-field.edge_nt, -19500, -19000, -18500, -field.core_nt, -17000])(values)
      .map((c) =>
        c.coordinates.flatMap((poly) =>
          poly.map((ring) => ring.map(([x, y]) => [field.lons[0] + (x - 0.5) * step, field.lats[0] + (y - 0.5) * step])),
        ),
      );
  }, [field]);

  // Live animation state kept in refs so the rAF loop never re-subscribes
  const live = useRef({ lastT: 0, stepAt: performance.now(), avgWall: 4, running: mission.running, total: 0, inSaa: false });
  const bursts = useRef<Burst[]>([]);
  const scars = useRef<Burst[]>([]);
  const seen = useRef(0);

  useEffect(() => {
    const L = live.current;
    L.running = mission.running;
    L.total = mission.config.total_steps * stepMin;
    if (steps.length < seen.current) {
      seen.current = 0; // new mission
      scars.current = [];
    }
    while (seen.current < steps.length) {
      const s = steps[seen.current++];
      const walls = steps.slice(-4).map((x) => x.wall_s);
      L.avgWall = Math.max(0.6, walls.reduce((a, b) => a + b, 0) / walls.length);
      L.lastT = s.t_min;
      L.stepAt = performance.now();
      L.inSaa = s.in_saa;
      strikeClusters(s).forEach((c, k) => {
        const [lon, lat] = positionAt(track, c.t);
        const b = {
          lon, lat, born: performance.now() + k * 140, n: Math.min(40, 6 + c.n / 4), crit: c.crit,
          seeds: Array.from({ length: 40 }, () => Math.random()),
        };
        bursts.current.push(b);
        scars.current.push(b);
      });
    }
    if (!steps.length) {
      scars.current = [];
      bursts.current = [];
      L.lastT = 0;
      L.stepAt = performance.now();
    }
  }, [steps, mission.running, mission.config.total_steps, stepMin, track]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    const land = landDots();
    let colors = { saa: "", grat: "", line: "", landC: "", accent: "", ink: "", ink4: "", crit: "" };
    let colorsAt = -1e9;

    const draw = (now: number) => {
      if (now - colorsAt > 1000) {
        colors = {
          saa: cssVar("--saa"), grat: cssVar("--graticule"), line: cssVar("--line-strong"), landC: cssVar("--land"),
          accent: cssVar("--accent"), ink: cssVar("--ink"), ink4: cssVar("--ink-4"), crit: cssVar("--crit"),
        };
        colorsAt = now;
      }
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const px = (lon: number) => ((lon + 180) / 360) * W;
      const py = (lat: number) => ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * H;

      ctx.strokeStyle = colors.grat;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let lon = -180; lon <= 180; lon += 30) {
        ctx.moveTo(px(lon), 0);
        ctx.lineTo(px(lon), H);
      }
      for (let lat = -60; lat <= 80; lat += 30) {
        ctx.moveTo(0, py(lat));
        ctx.lineTo(W, py(lat));
      }
      ctx.stroke();

      // SAA bands breathe slightly so the hazard reads as "live"
      const breathe = 0.85 + 0.15 * Math.sin(now / 900);
      bands.forEach((rings, k) => {
        ctx.fillStyle = `rgba(${colors.saa}, ${(0.06 + k * 0.05) * breathe})`;
        ctx.beginPath();
        for (const ring of rings) {
          ring.forEach(([lon, lat], i) => (i ? ctx.lineTo(px(lon), py(lat)) : ctx.moveTo(px(lon), py(lat))));
          ctx.closePath();
        }
        ctx.fill("evenodd");
      });

      ctx.fillStyle = colors.landC;
      const r = Math.max(1.3, W / 560);
      for (const [lon, lat] of land) {
        if (lat > LAT_MAX || lat < LAT_MIN) continue;
        ctx.fillRect(px(lon) - r / 2, py(lat) - r / 2, r, r);
      }

      // Satellite time: glide from the last completed step toward the next one
      const L = live.current;
      const frac = L.running ? Math.min(0.97, (now - L.stepAt) / 1000 / L.avgWall) : 0;
      const minutes = Math.min(L.total, L.lastT + frac * stepMin);

      const path = (from: number, to: number, n: number, style: (f: number) => void) => {
        let prev: [number, number] | null = null;
        for (let k = 0; k <= n; k++) {
          const m = from + ((to - from) * k) / n;
          if (m < 0) continue;
          const p = positionAt(track, m);
          if (prev && Math.abs(p[0] - prev[0]) < 180) {
            style(k / n);
            ctx.beginPath();
            ctx.moveTo(px(prev[0]), py(prev[1]));
            ctx.lineTo(px(p[0]), py(p[1]));
            ctx.stroke();
          }
          prev = p;
        }
      };
      ctx.setLineDash([2, 4]);
      path(minutes, minutes + track.period_min, 140, () => {
        ctx.strokeStyle = colors.ink4;
        ctx.globalAlpha = 0.6;
        ctx.lineWidth = 1;
      });
      ctx.setLineDash([]);
      path(minutes - TRAIL_MIN, minutes, 400, (f) => {
        ctx.strokeStyle = colors.accent;
        ctx.globalAlpha = f ** 2;
        ctx.lineWidth = 0.6 + 2 * f;
      });
      ctx.globalAlpha = 1;

      // Strike scars (where this mission has been hit), then the fresh bursts on top
      for (const b of scars.current) {
        ctx.fillStyle = b.crit ? colors.crit : `rgb(${colors.saa})`;
        ctx.globalAlpha = b.crit ? 0.75 : 0.4;
        ctx.beginPath();
        ctx.arc(px(b.lon), py(b.lat), b.crit ? 2.6 : 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      bursts.current = bursts.current.filter((b) => now - b.born < 2200);
      for (const b of bursts.current) {
        if (now < b.born) continue; // staggered along the step's path
        const age = (now - b.born) / 2200;
        const x = px(b.lon);
        const y = py(b.lat);
        ctx.strokeStyle = b.crit ? colors.crit : `rgb(${colors.saa})`;
        ctx.globalAlpha = (1 - age) * 0.9;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 6 + age * 46, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        for (let i = 0; i < b.n; i++) {
          const ang = b.seeds[i] * Math.PI * 2;
          const dist = (8 + b.seeds[(i + 7) % 40] * 38) * Math.min(1, age * 2.2);
          ctx.globalAlpha = (1 - age) * 0.95;
          ctx.fillRect(x + Math.cos(ang) * dist - 1, y + Math.sin(ang) * dist - 1, 2, 2);
        }
      }
      ctx.globalAlpha = 1;

      const [lon, lat] = positionAt(track, minutes);
      const x = px(lon);
      const y = py(lat);
      const hot = L.inSaa && L.running;
      const pulse = 22 + 6 * Math.sin(now / 260);
      const g = ctx.createRadialGradient(x, y, 0, x, y, pulse);
      g.addColorStop(0, hot ? `rgba(${colors.saa}, 0.6)` : colors.accent + "77");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, pulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = colors.ink;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = hot ? `rgb(${colors.saa})` : colors.accent;
      ctx.lineWidth = 2;
      ctx.stroke();

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [bands, track, stepMin, theme]);

  return <canvas ref={ref} className="live-map" role="img" aria-label="Live ground track over the South Atlantic Anomaly" />;
}

/** Groups a step's strike minutes into bursts (gaps over 6 min split a cluster). Steps from
 * engines without per-hit minutes fall back to one burst at the end of the step. */
function strikeClusters(s: MissionStep): { t: number; n: number; crit: boolean }[] {
  if (s.flips === 0) return [];
  if (!s.hit_minutes?.length) return [{ t: s.t_min, n: s.flips, crit: s.msb > 0 }];
  const critAt = s.hits.filter((h) => h.bit === 30 && h.t_min !== undefined).map((h) => h.t_min as number);
  const sorted = [...s.hit_minutes].sort((a, b) => a - b);
  const out: { t: number; n: number; crit: boolean }[] = [];
  let group: number[] = [];
  const flush = () => {
    if (!group.length) return;
    const lo = group[0];
    const hi = group[group.length - 1];
    out.push({ t: group.reduce((a, b) => a + b, 0) / group.length, n: group.length, crit: critAt.some((m) => m >= lo && m <= hi) });
    group = [];
  };
  for (const m of sorted) {
    if (group.length && m - group[group.length - 1] > 6) flush();
    group.push(m);
  }
  flush();
  return out;
}
