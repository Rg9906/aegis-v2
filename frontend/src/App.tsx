import { useEffect, useState } from "react";
import type { Campaign } from "./lib/types";
import { isPreFix, loadCampaign, loadIndex } from "./lib/data";
import { useTheme } from "./lib/useMissionClock";
import Nav from "./components/Nav";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import Replay from "./components/replay/Replay";
import Pareto from "./components/Pareto";
import Scoreboard from "./components/Scoreboard";
import Incidents from "./components/Incidents";
import Method from "./components/Method";

/** Prefer the newest "final" campaign, then any GPU run, then whatever exists. */
function pickDefault(names: string[]) {
  return (
    names.find((n) => n === "main_final") ??
    names.find((n) => n.startsWith("main_final")) ??
    names.find((n) => n.startsWith("main")) ??
    names[0]
  );
}

export default function App() {
  const { theme, setTheme } = useTheme();
  const [campaigns, setCampaigns] = useState<string[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadIndex()
      .then((names) => {
        setCampaigns(names);
        const fromUrl = new URLSearchParams(location.search).get("campaign");
        setCurrent(fromUrl && names.includes(fromUrl) ? fromUrl : pickDefault(names));
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!current) return;
    setCampaign(null);
    loadCampaign(current)
      .then(setCampaign)
      .catch((e) => setError(String(e)));
  }, [current]);

  if (error) {
    return (
      <div className="state-screen">
        <h1>No mission data</h1>
        <p>
          Export a campaign first: <code>python experiments/export_frontend_data.py --name main</code>
        </p>
        <p className="dim mono">{error}</p>
      </div>
    );
  }
  if (!campaign || !current) {
    return (
      <div className="state-screen" aria-busy="true">
        <div className="loader" />
        <p className="dim mono">Acquiring telemetry…</p>
      </div>
    );
  }

  return (
    <>
      <div className="grain" aria-hidden="true" />
      <Nav campaigns={campaigns} current={current} onCampaign={setCurrent} theme={theme} onTheme={setTheme} />
      <main>
        <Hero campaign={campaign} theme={theme} />
        <HowItWorks />
        <Replay key={current} campaign={campaign} theme={theme} />
        <Pareto campaign={campaign} />
        <Scoreboard campaign={campaign} />
        <Incidents preFix={isPreFix(campaign)} />
        <Method campaign={campaign} />
      </main>
      <footer className="footer">
        <div>
          <span className="brand-name">AEGIS</span> <span className="mono dim">v2 · research prototype</span>
        </div>
        <div className="mono dim">
          campaign {campaign.name} · {campaign.setup.arch} · exported {campaign.generated.slice(0, 10)}
        </div>
      </footer>
    </>
  );
}
