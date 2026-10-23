import { lazy, Suspense, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import App from "./App";
import type { Mode } from "./components/ModeSwitch";

const Console = lazy(() => import("./console/Console"));

const modeOf = (hash: string): Mode => (hash.startsWith("#/console") ? "console" : "briefing");

/**
 * Two modes in one app: the scrolling Briefing (explanation deck, recorded campaigns) and the
 * live Console (real engine). Briefing section anchors (#replay ...) never start with "#/".
 */
export default function Root() {
  const [mode, setMode] = useState<Mode>(() => modeOf(location.hash));

  useEffect(() => {
    const onHash = () => {
      const next = modeOf(location.hash);
      setMode((prev) => {
        if (prev !== next) window.scrollTo({ top: 0 });
        return next;
      });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.title = mode === "console" ? "Aegis-v2 · Live Mission Console" : "Aegis-v2 Mission Dossier";
    document.documentElement.dataset.mode = mode;
  }, [mode]);

  return (
    <>
      <AnimatePresence mode="wait" initial={false}>
        {/* Opacity only: a filter/transform here would re-anchor the Briefing's fixed/sticky layers */}
        <motion.div
          key={mode}
          className="mode-root"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35, ease: [0.2, 0.7, 0.1, 1] }}
        >
          {mode === "console" ? (
            <Suspense
              fallback={
                <div className="state-screen" aria-busy="true">
                  <div className="loader" />
                </div>
              }
            >
              <Console />
            </Suspense>
          ) : (
            <App />
          )}
        </motion.div>
      </AnimatePresence>
      <AnimatePresence>
        <motion.div
          key={`warp-${mode}`}
          className="mode-warp"
          aria-hidden="true"
          initial={{ scaleX: 0, opacity: 1 }}
          animate={{ scaleX: 1, opacity: 0 }}
          transition={{ duration: 0.9, ease: [0.2, 0.7, 0.1, 1] }}
        />
      </AnimatePresence>
    </>
  );
}
