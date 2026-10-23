import { useEffect, useRef, useState } from "react";
import { useEngine } from "./engine";

const LEVELS = ["all", "alerts"] as const;

/** Terminal-style flight log streamed from the engine; follows the tail unless scrolled up. */
export default function EventLog({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { logs } = useEngine();
  const [filter, setFilter] = useState<(typeof LEVELS)[number]>("all");
  const box = useRef<HTMLDivElement>(null);
  const follow = useRef(true);

  const shown = filter === "all" ? logs : logs.filter((l) => l.level === "crit" || l.level === "warn" || l.level === "error");

  useEffect(() => {
    const el = box.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  }, [shown.length, open]);

  const alerts = logs.filter((l) => l.level === "crit" || l.level === "error").length;

  return (
    <aside className={`c-log ${open ? "open" : ""}`} aria-label="Flight log">
      <div className="c-log-head">
        <button className="c-log-toggle" onClick={onToggle} aria-expanded={open} title="Toggle log (L)">
          <span className="c-log-caret" aria-hidden="true">
            ▸
          </span>
          Flight log
          <span className="mono dim">{logs.length}</span>
          {alerts > 0 && <span className="c-log-alerts mono">{alerts} alerts</span>}
        </button>
        {open && (
          <div className="seg seg-sm" role="group" aria-label="Log filter">
            {LEVELS.map((l) => (
              <button key={l} className={filter === l ? "on" : ""} onClick={() => setFilter(l)}>
                {l}
              </button>
            ))}
          </div>
        )}
      </div>
      {open && (
        <div
          className="c-log-body mono"
          ref={box}
          onScroll={(e) => {
            const el = e.currentTarget;
            follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {shown.length === 0 && <p className="dim">Waiting for telemetry…</p>}
          {shown.map((l, i) => (
            <div key={`${l.t}-${i}`} className={`log-line log-${l.level}`}>
              <span className="log-time">{new Date(l.t * 1000).toLocaleTimeString([], { hour12: false })}</span>
              <span className="log-tag">{l.tag}</span>
              <span className="log-text">{l.text}</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
