export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const pts = (x: number, digits = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(digits)}`;
export const mb = (x: number) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

/** Mission clock: "T+1d 04:00" from elapsed minutes. */
export function missionClock(minutes: number) {
  const d = Math.floor(minutes / 1440);
  const h = Math.floor((minutes % 1440) / 60);
  const m = Math.floor(minutes % 60);
  return `T+${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export const latLon = (lat: number, lon: number) =>
  `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? "N" : "S"}  ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? "E" : "W"}`;

/** Readable block names: layer3_block0 -> "L3·0", features_block7 -> "F7". */
export function blockShort(id: string) {
  if (id === "stem") return "STEM";
  if (id === "classifier") return "FC";
  const m = id.match(/^layer(\d)_block(\d)$/);
  if (m) return `L${m[1]}·${m[2]}`;
  const f = id.match(/^features_block(\d+)$/);
  if (f) return `F${f[1]}`;
  return id;
}

export const sci = (x: number) => {
  const [m, e] = x.toExponential(2).split("e");
  return `${m}×10${e.replace("+", "").replace(/^-/, "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)])}`;
};
