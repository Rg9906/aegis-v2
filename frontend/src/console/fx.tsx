import { useEffect, useRef, useState } from "react";
import { animate } from "motion/react";
import { cssVar, usePrefersReducedMotion } from "../lib/useMissionClock";
import type { Num } from "./types";

/** Tweens to each new value (tabular digits, so the layout never jitters). */
export function AnimatedNumber({
  value,
  decimals = 1,
  suffix = "",
  duration = 0.9,
  className = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const reduced = usePrefersReducedMotion();
  useEffect(() => {
    if (reduced) {
      setShown(value);
      from.current = value;
      return;
    }
    const c = animate(from.current, value, {
      duration,
      ease: [0.2, 0.7, 0.1, 1],
      onUpdate: (v) => {
        from.current = v;
        setShown(v);
      },
    });
    return () => c.stop();
  }, [value, duration, reduced]);
  return (
    <span className={`tabular ${className}`}>
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  );
}

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** Human form of a stored float: 0.0213, -1.03×10³⁸, NaN, ∞. */
export function fmtValue(x: Num): string {
  if (x === "NaN") return "NaN";
  if (x === "Infinity") return "+∞";
  if (x === "-Infinity") return "−∞";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) {
    const [m, e] = x.toExponential(2).split("e");
    const exp = Number(e);
    const sup = (exp < 0 ? "⁻" : "") + String(Math.abs(exp)).replace(/\d/g, (d) => SUP[Number(d)]);
    return `${m.replace("-", "−")}×10${sup}`;
  }
  return x.toPrecision(4).replace("-", "−");
}

/** Magnitude change caused by a flip, as a factor string ("×3.4×10³⁸", "÷2", "NaN"). */
export function fmtFactor(from: Num, to: Num): string {
  if (typeof from !== "number" || typeof to !== "number") return "non-finite";
  if (from === 0) return to === 0 ? "×1" : "from 0";
  const r = Math.abs(to / from);
  if (r >= 1) return `×${r >= 1e4 ? fmtValue(r) : r.toPrecision(3)}`;
  return `÷${1 / r >= 1e4 ? fmtValue(1 / r) : (1 / r).toPrecision(3)}`;
}

export const pct1 = (x: number | null | undefined) => (x == null ? "—" : `${(100 * x).toFixed(1)}%`);
export const mbStr = (bytes: number) => {
  const m = bytes / 1e6;
  return m >= 100 ? m.toFixed(0) : m >= 10 ? m.toFixed(1) : m.toFixed(2);
};

/** Slowly drifting starfield with occasional streaks; canvas, theme-aware, pauses when hidden. */
export function Starfield() {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let W = 0;
    let H = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const stars = Array.from({ length: 220 }, () => ({
      x: Math.random(),
      y: Math.random(),
      z: 0.2 + Math.random() * 0.8,
      tw: Math.random() * Math.PI * 2,
    }));
    const resize = () => {
      W = canvas.clientWidth;
      H = canvas.clientHeight;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    let streak: { x: number; y: number; vx: number; vy: number; life: number } | null = null;
    // getComputedStyle every frame is costly: refresh theme colors once a second
    let ink = cssVar("--ink");
    let accent = cssVar("--accent");
    let colorsAt = 0;
    const draw = (now: number) => {
      ctx.clearRect(0, 0, W, H);
      if (now - colorsAt > 1000) {
        ink = cssVar("--ink");
        accent = cssVar("--accent");
        colorsAt = now;
      }
      for (const s of stars) {
        if (!reduced) s.x = (s.x + 0.000012 * s.z) % 1;
        const a = 0.18 + 0.5 * s.z * (0.6 + 0.4 * Math.sin(now / 900 + s.tw));
        ctx.globalAlpha = a * 0.55;
        ctx.fillStyle = ink;
        const r = 0.4 + s.z * 0.9;
        ctx.fillRect(s.x * W, s.y * H, r, r);
      }
      if (!reduced) {
        if (!streak && Math.random() < 0.004) {
          streak = { x: Math.random() * W, y: Math.random() * H * 0.5, vx: 7 + Math.random() * 5, vy: 2 + Math.random() * 2, life: 1 };
        }
        if (streak) {
          const g = ctx.createLinearGradient(streak.x, streak.y, streak.x - streak.vx * 14, streak.y - streak.vy * 14);
          g.addColorStop(0, accent);
          g.addColorStop(1, "transparent");
          ctx.globalAlpha = streak.life * 0.8;
          ctx.strokeStyle = g;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(streak.x, streak.y);
          ctx.lineTo(streak.x - streak.vx * 14, streak.y - streak.vy * 14);
          ctx.stroke();
          streak.x += streak.vx;
          streak.y += streak.vy;
          streak.life -= 0.02;
          if (streak.life <= 0) streak = null;
        }
      }
      ctx.globalAlpha = 1;
      if (!document.hidden) raf = requestAnimationFrame(draw);
    };
    const onVis = () => {
      if (!document.hidden) raf = requestAnimationFrame(draw);
    };
    document.addEventListener("visibilitychange", onVis);
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [reduced]);
  return <canvas ref={ref} className="starfield" aria-hidden="true" />;
}
