import { useEffect, useMemo, useRef } from "react";
import { contours } from "d3-contour";
import type { Campaign } from "../../lib/types";
import { landDots, positionAt } from "../../lib/geo";
import { cssVar, useElementWidth } from "../../lib/useMissionClock";

const LAT_MAX = 84;
const LAT_MIN = -84;
const TRAIL_MIN = 300; // five hours of trail

interface Props {
  campaign: Campaign;
  minutes: number;
  theme: string;
  inSaa: boolean;
}

/** Equirectangular ground map: halftone land, SAA isobands, fading ground track, satellite. */
export default function GroundMap({ campaign, minutes, theme, inSaa }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(ref);
  const track = campaign.track;
  const field = campaign.field;

  // SAA isobands between the ramp edge (20,000 nT) and core (18,000 nT), in lon/lat rings
  const bands = useMemo(() => {
    if (!field) return [];
    const nx = field.lons.length;
    const ny = field.lats.length;
    const step = field.lons[1] - field.lons[0];
    const values = field.lats.flatMap((_, i) => field.lons.map((__, j) => -field.b_nt[i][j]));
    const levels = [-field.edge_nt, -19500, -19000, -18500, -field.core_nt, -17000];
    return contours()
      .size([nx, ny])
      .thresholds(levels)(values)
      .map((c) => ({
        level: c.value,
        rings: c.coordinates.flatMap((poly) =>
          poly.map((ring) => ring.map(([x, y]) => [field.lons[0] + (x - 0.5) * step, field.lats[0] + (y - 0.5) * step])),
        ),
      }));
  }, [field]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
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
    const saa = cssVar("--saa");

    // Graticule
    ctx.strokeStyle = cssVar("--graticule");
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
    ctx.strokeStyle = cssVar("--line-strong");
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, py(0));
    ctx.lineTo(W, py(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // SAA isobands (planar fill; rings are already in lon/lat)
    bands.forEach((band, k) => {
      ctx.fillStyle = `rgba(${saa}, ${0.07 + k * 0.05})`;
      ctx.beginPath();
      for (const ring of band.rings) {
        ring.forEach(([lon, lat], i) => (i ? ctx.lineTo(px(lon), py(lat)) : ctx.moveTo(px(lon), py(lat))));
        ctx.closePath();
      }
      ctx.fill("evenodd");
    });
    if (bands.length) {
      ctx.strokeStyle = `rgba(${saa}, 0.55)`;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      for (const ring of bands[0].rings) {
        ring.forEach(([lon, lat], i) => (i ? ctx.lineTo(px(lon), py(lat)) : ctx.moveTo(px(lon), py(lat))));
        ctx.closePath();
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Halftone land
    ctx.fillStyle = cssVar("--land");
    const r = Math.max(1.4, W / 520);
    for (const [lon, lat] of landDots()) {
      if (lat > LAT_MAX || lat < LAT_MIN) continue;
      ctx.fillRect(px(lon) - r / 2, py(lat) - r / 2, r, r);
    }

    if (!track) return;
    const accent = cssVar("--accent");
    // Future: one orbit ahead, faint dashed
    const period = track.period_min;
    const drawPath = (from: number, to: number, n: number, style: (f: number) => void) => {
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
    drawPath(minutes, minutes + period, 160, () => {
      ctx.strokeStyle = cssVar("--ink-4");
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1;
    });
    ctx.setLineDash([]);
    drawPath(minutes - TRAIL_MIN, minutes, 600, (f) => {
      ctx.strokeStyle = accent;
      ctx.globalAlpha = f ** 2;
      ctx.lineWidth = 0.6 + 1.6 * f;
    });
    ctx.globalAlpha = 1;

    const [lon, lat] = positionAt(track, minutes);
    const x = px(lon);
    const y = py(lat);
    const g = ctx.createRadialGradient(x, y, 0, x, y, 26);
    g.addColorStop(0, inSaa ? `rgba(${saa}, 0.55)` : accent + "66");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = cssVar("--ink");
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = inSaa ? `rgb(${saa})` : accent;
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [bands, track, minutes, theme, inSaa, width]);

  return (
    <canvas
      ref={ref}
      className="ground-map"
      role="img"
      aria-label="World map with the satellite ground track and the South Atlantic Anomaly"
    />
  );
}
