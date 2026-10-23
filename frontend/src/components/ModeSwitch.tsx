import { motion } from "motion/react";

export type Mode = "briefing" | "console";

/** Briefing <-> Console pill; the active half slides between the two. */
export default function ModeSwitch({ mode }: { mode: Mode }) {
  const items: { id: Mode; label: string; href: string }[] = [
    { id: "briefing", label: "Briefing", href: "#overview" },
    { id: "console", label: "Live console", href: "#/console" },
  ];
  return (
    <div className="mode-switch" role="tablist" aria-label="Mode">
      {items.map((it) => (
        <a key={it.id} href={it.href} role="tab" aria-selected={mode === it.id} className={mode === it.id ? "on" : ""}>
          {mode === it.id && (
            <motion.span layoutId="mode-pill" className="mode-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />
          )}
          <span className="mode-label">
            {it.id === "console" && <span className={`mode-live ${mode === "console" ? "on" : ""}`} aria-hidden="true" />}
            {it.label}
          </span>
        </a>
      ))}
    </div>
  );
}
