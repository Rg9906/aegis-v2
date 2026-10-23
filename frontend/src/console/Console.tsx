import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import ModeSwitch from "../components/ModeSwitch";
import { useTheme } from "../lib/useMissionClock";
import { EngineProvider, useEngine } from "./engine";
import { Starfield } from "./fx";
import BootScreen from "./BootScreen";
import EventLog from "./EventLog";
import Lab from "./lab/Lab";
import Mission from "./mission/Mission";
import Inspector from "./inspect/Inspector";
import "../styles/console.css";

type Workspace = "lab" | "mission" | "inspect";
const WORKSPACES: { id: Workspace; label: string; hint: string; key: string }[] = [
  { id: "lab", label: "Strike Lab", hint: "Fire radiation at chosen bits, watch detection and repair", key: "1" },
  { id: "mission", label: "Live Mission", hint: "Fly the real Sentinel-2A orbit; four strategies race", key: "2" },
  { id: "inspect", label: "Inspector", hint: "Classify real satellite chips with the live payload", key: "3" },
];

const workspaceOf = (hash: string): Workspace => {
  const w = hash.split("/")[2];
  return w === "mission" || w === "inspect" ? w : "lab";
};

export default function Console() {
  return (
    <EngineProvider>
      <ConsoleShell />
    </EngineProvider>
  );
}

function ConsoleShell() {
  const { connection, snapshot, toasts, dismissToast, mission } = useEngine();
  const { theme, setTheme } = useTheme();
  const [ws, setWs] = useState<Workspace>(() => workspaceOf(location.hash));
  const [logOpen, setLogOpen] = useState(true);

  useEffect(() => {
    const onHash = () => setWs(workspaceOf(location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // 1/2/3 switch workspaces, L toggles the log (ignored while typing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement;
      if (e.metaKey || e.ctrlKey || e.altKey || (el && ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName))) return;
      const w = WORKSPACES.find((x) => x.key === e.key);
      if (w) location.hash = `#/console/${w.id}`;
      if (e.key.toLowerCase() === "l") setLogOpen((o) => !o);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const ready = connection === "online" && snapshot?.status === "ready";
  const missionLive = !!mission?.running;

  return (
    <div className="console" data-ws={ws}>
      <Starfield />
      <div className="console-vignette" aria-hidden="true" />
      <header className="c-header">
        <a className="brand" href="#overview" aria-label="Aegis-v2 briefing">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="16" cy="16" r="9" fill="none" stroke="var(--accent)" strokeWidth="1.8" />
            <ellipse cx="16" cy="16" rx="14" ry="4.8" fill="none" stroke="currentColor" strokeWidth="1.3" transform="rotate(-24 16 16)" />
            <circle cx="27.2" cy="10.9" r="2.1" fill="currentColor" className="brand-sat" />
          </svg>
          <span className="brand-name">AEGIS</span>
          <span className="brand-ver mono">v2</span>
        </a>
        <ModeSwitch mode="console" />
        <nav className="c-tabs" aria-label="Workspaces">
          {WORKSPACES.map((w) => (
            <a key={w.id} href={`#/console/${w.id}`} className={ws === w.id ? "on" : ""} title={`${w.hint} (key ${w.key})`}>
              {ws === w.id && <motion.span layoutId="c-tab-pill" className="c-tab-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
              <span className="c-tab-label">
                {w.label}
                {w.id === "mission" && missionLive && <span className="c-tab-live" aria-label="running" />}
              </span>
            </a>
          ))}
        </nav>
        <div className="c-header-tools">
          <ArchSwitch />
          <EnginePill />
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? (
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" /><g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.1 1.1M11.9 11.9 13 13M3 13l1.1-1.1M11.9 4.1 13 3" /></g></svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 9.6A5.8 5.8 0 0 1 6.4 2.5a5.8 5.8 0 1 0 7.1 7.1Z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" /></svg>
            )}
          </button>
        </div>
      </header>

      <div className={`c-body ${logOpen ? "log-open" : ""}`}>
        <main className="c-main">
          {ready ? (
            <AnimatePresence mode="wait">
              <motion.div
                key={ws}
                className="c-workspace"
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.32, ease: [0.2, 0.7, 0.1, 1] }}
              >
                {ws === "lab" && <Lab />}
                {ws === "mission" && <Mission />}
                {ws === "inspect" && <Inspector />}
              </motion.div>
            </AnimatePresence>
          ) : (
            <BootScreen />
          )}
        </main>
        <EventLog open={logOpen} onToggle={() => setLogOpen((o) => !o)} />
      </div>

      <div className="toasts" role="status" aria-live="polite">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.button
              key={t.id}
              className={`toast toast-${t.level}`}
              onClick={() => dismissToast(t.id)}
              initial={{ opacity: 0, x: 40, scale: 0.96 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 40 }}
              layout
            >
              {t.text}
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

function EnginePill() {
  const { connection, snapshot, busy } = useEngine();
  const state =
    connection !== "online" ? connection : snapshot?.status === "ready" ? (busy ? "busy" : "ready") : snapshot?.status ?? "connecting";
  const label: Record<string, string> = {
    connecting: "Linking",
    offline: "Engine offline",
    ready: "Engine online",
    busy: "Computing",
    booting: "Booting",
    error: "Engine error",
    idle: "Idle",
  };
  return (
    <div className={`engine-pill engine-${state}`} title={snapshot?.threads ? `${snapshot.threads} PyTorch CPU threads` : undefined}>
      <span className="engine-dot" aria-hidden="true" />
      <span>{label[state] ?? state}</span>
      {snapshot?.status === "ready" && <span className="engine-meta mono">{snapshot.threads} thr</span>}
    </div>
  );
}

function ArchSwitch() {
  const { snapshot, run, mission } = useEngine();
  if (!snapshot?.archs || snapshot.archs.length < 2) return null;
  const disabled = snapshot.status !== "ready" || !!mission?.running;
  return (
    <div className="seg seg-sm arch-switch" role="group" aria-label="Payload architecture">
      {snapshot.archs.map((a) => (
        <button
          key={a}
          className={snapshot.arch === a ? "on" : ""}
          disabled={disabled || snapshot.arch === a}
          onClick={() => run("Switch architecture", "arch", { arch: a })}
          title={disabled ? "Unavailable while the engine is busy" : `Reload the payload as ${a}`}
        >
          {a === "mobilenet_v2" ? "MobileNetV2" : "ResNet-18"}
        </button>
      ))}
    </div>
  );
}
