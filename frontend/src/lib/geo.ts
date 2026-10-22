import { geoEquirectangular, geoPath, type GeoPermissibleObjects } from "d3-geo";
import { feature } from "topojson-client";
import type { Topology, GeometryCollection } from "topojson-specification";
import landTopo from "world-atlas/land-110m.json";
import type { Campaign } from "./types";

export const LAND = feature(
  landTopo as unknown as Topology,
  (landTopo as unknown as Topology).objects.land as GeometryCollection,
) as GeoPermissibleObjects;

let cachedDots: [number, number][] | null = null;

/**
 * Halftone land: [lon, lat] points on a near-equal-area lattice, kept where land is.
 * Land membership comes from rasterizing the polygons once to an offscreen canvas.
 */
export function landDots(spacingDeg = 1.6): [number, number][] {
  if (cachedDots) return cachedDots;
  const W = 1440, H = 720;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const proj = geoEquirectangular().scale(W / (2 * Math.PI)).translate([W / 2, H / 2]);
  ctx.fillStyle = "#000";
  ctx.beginPath();
  geoPath(proj, ctx)(LAND);
  ctx.fill();
  const px = ctx.getImageData(0, 0, W, H).data;
  const dots: [number, number][] = [];
  for (let lat = -58; lat <= 84; lat += spacingDeg) {
    const step = spacingDeg / Math.max(Math.cos((lat * Math.PI) / 180), 0.2);
    // Offset alternate rows for a hexagonal look
    const offset = (Math.round(lat / spacingDeg) % 2) * step * 0.5;
    for (let lon = -180 + offset; lon < 180; lon += step) {
      const [x, y] = proj([lon, lat])!;
      const i = (Math.floor(y) * W + Math.floor(x)) * 4 + 3;
      if (px[i] > 128) dots.push([lon, lat]);
    }
  }
  cachedDots = dots;
  return dots;
}

/** Satellite position at `minutes` after epoch, interpolated along the exported track. */
export function positionAt(track: NonNullable<Campaign["track"]>, minutes: number): [number, number] {
  const { minutes: ms, lat, lon } = track;
  const dt = ms[1] - ms[0];
  const i = Math.max(0, Math.min(ms.length - 2, Math.floor(minutes / dt)));
  const f = Math.max(0, Math.min(1, (minutes - ms[i]) / dt));
  let dLon = lon[i + 1] - lon[i];
  if (dLon > 180) dLon -= 360;
  if (dLon < -180) dLon += 360;
  let lo = lon[i] + f * dLon;
  if (lo > 180) lo -= 360;
  if (lo < -180) lo += 360;
  return [lo, lat[i] + f * (lat[i + 1] - lat[i])];
}

/** Track points within [from, to] minutes, split where the path wraps the antimeridian. */
export function trackSegments(track: NonNullable<Campaign["track"]>, from: number, to: number) {
  const segs: [number, number][][] = [];
  let cur: [number, number][] = [];
  for (let i = 0; i < track.minutes.length; i++) {
    const m = track.minutes[i];
    if (m < from || m > to) continue;
    const p: [number, number] = [track.lon[i], track.lat[i]];
    if (cur.length && Math.abs(p[0] - cur[cur.length - 1][0]) > 180) {
      segs.push(cur);
      cur = [];
    }
    cur.push(p);
  }
  if (cur.length) segs.push(cur);
  return segs;
}

/** 0..1 SAA weight from |B| using the campaign's linear ramp (edge -> core). */
export function saaWeight(bNt: number, edge: number, core: number) {
  return Math.max(0, Math.min(1, (edge - bNt) / (edge - core)));
}
