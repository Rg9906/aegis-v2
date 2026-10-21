import { useEffect, useState } from "react";

const LINKS = [
  ["how", "How it works"],
  ["replay", "Replay"],
  ["tradeoff", "Trade-off"],
  ["scoreboard", "Scoreboard"],
  ["incidents", "Incidents"],
  ["method", "Method"],
] as const;

interface Props {
  campaigns: string[];
  current: string;
  onCampaign: (name: string) => void;
  theme: "dark" | "light";
  onTheme: (t: "dark" | "light") => void;
}

export default function Nav({ campaigns, current, onCampaign, theme, onTheme }: Props) {
  const [active, setActive] = useState<string>("");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 12);
      if (window.scrollY < window.innerHeight * 0.5) setActive(""); // hero: no section highlighted
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActive(e.target.id)),
      { rootMargin: "-45% 0px -50% 0px" },
    );
    LINKS.forEach(([id]) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });
    return () => {
      window.removeEventListener("scroll", onScroll);
      io.disconnect();
    };
  }, [current]);

  return (
    <nav className={`nav ${scrolled ? "nav-scrolled" : ""}`} aria-label="Sections">
      <a className="brand" href="#overview" aria-label="Aegis-v2, back to top">
        <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
          <circle cx="16" cy="16" r="9" fill="none" stroke="var(--accent)" strokeWidth="1.8" />
          <ellipse cx="16" cy="16" rx="14" ry="4.8" fill="none" stroke="currentColor" strokeWidth="1.3" transform="rotate(-24 16 16)" />
          <circle cx="27.2" cy="10.9" r="2.1" fill="currentColor" />
        </svg>
        <span className="brand-name">AEGIS</span>
        <span className="brand-ver mono">v2</span>
      </a>
      <div className="nav-links">
        {LINKS.map(([id, label]) => (
          <a key={id} href={`#${id}`} className={active === id ? "on" : ""}>
            {label}
          </a>
        ))}
      </div>
      <div className="nav-tools">
        {campaigns.length > 1 && (
          <label className="select">
            <span className="sr-only">Campaign</span>
            <select value={current} onChange={(e) => onCampaign(e.target.value)}>
              {campaigns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        )}
        <button
          className="icon-btn"
          onClick={() => onTheme(theme === "dark" ? "light" : "dark")}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? (
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" /><g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.1 1.1M11.9 11.9 13 13M3 13l1.1-1.1M11.9 4.1 13 3" /></g></svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 9.6A5.8 5.8 0 0 1 6.4 2.5a5.8 5.8 0 1 0 7.1 7.1Z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" /></svg>
          )}
        </button>
      </div>
    </nav>
  );
}
