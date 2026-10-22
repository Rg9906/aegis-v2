import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

/** Simulated hours advanced per real second. */
export const SPEEDS = [0.5, 1, 3, 8] as const;

/**
 * Mission playback clock in simulated minutes, driven by requestAnimationFrame.
 * Stops at the end of the mission; `seek` jumps anywhere.
 */
export function useMissionClock(totalMinutes: number) {
  const [minutes, setMinutes] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(3);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      if (last.current !== null) {
        const dt = (now - last.current) / 1000;
        setMinutes((m) => {
          const next = m + dt * speed * 60;
          if (next >= totalMinutes) {
            setPlaying(false);
            return totalMinutes - 1e-6;
          }
          return next;
        });
      }
      last.current = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, totalMinutes]);

  const seek = useCallback(
    (m: number) => setMinutes(Math.max(0, Math.min(totalMinutes - 1e-6, m))),
    [totalMinutes],
  );

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p && minutes >= totalMinutes - 1) setMinutes(0); // replay from the start
      return !p;
    });
  }, [minutes, totalMinutes]);

  return { minutes, playing, setPlaying, toggle, speed, setSpeed, seek };
}

/** Tracks the color scheme the app is rendering in (data-theme on <html>). */
export function useTheme() {
  const read = () =>
    (document.documentElement.dataset.theme as "dark" | "light" | undefined) ?? "dark";
  const [theme, setThemeState] = useState<"dark" | "light">(read);
  const setTheme = useCallback((t: "dark" | "light") => {
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem("aegis-theme", t);
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
    setThemeState(t);
  }, []);
  return { theme, setTheme };
}

/** Live width of an element (for responsive SVG / canvas drawing). */
export function useElementWidth(ref: RefObject<Element | null>) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

/** Reads a CSS custom property (series colors etc.) for canvas drawing. */
export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}
