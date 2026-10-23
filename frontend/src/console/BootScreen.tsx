import { motion } from "motion/react";
import { useEngine } from "./engine";

const STAGES = [
  { at: 0.05, text: "Load EuroSAT weights" },
  { at: 0.15, text: "Load Sentinel-2 chips (disjoint calibration / evaluation)" },
  { at: 0.3, text: "Partition into restorable blocks · compress golden store" },
  { at: 0.4, text: "Calibrate T1 activation envelope + T2 BN drift" },
  { at: 0.85, text: "Compute T3 CRC-32 references · measure clean accuracy" },
  { at: 0.95, text: "Evaluate IGRF field for the South Atlantic Anomaly" },
];

/** Shown until the engine is connected and ready: offline instructions, boot progress or error. */
export default function BootScreen() {
  const { connection, snapshot, boot } = useEngine();

  if (connection !== "online") {
    return (
      <div className="boot">
        <motion.div className="boot-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="boot-orbit" aria-hidden="true">
            <span className="boot-sat" />
          </div>
          <p className="overline">{connection === "connecting" ? "Establishing link" : "No downlink"}</p>
          <h2>
            {connection === "connecting" ? (
              <>
                Linking to the <em>flight engine</em>…
              </>
            ) : (
              <>
                The flight engine is <em>offline.</em>
              </>
            )}
          </h2>
          <p className="dim">
            The console drives the real PyTorch model, fault injector and Aegis-v2 monitors in a local Python process. Start it
            from the project folder; this page reconnects on its own.
          </p>
          <pre className="boot-cmd mono">
            <span className="dim">PS&gt; </span>.venv\Scripts\Activate.ps1{"\n"}
            <span className="dim">PS&gt; </span>python -m aegis.console
          </pre>
          <a className="btn" href="#overview">
            Open the briefing instead
          </a>
        </motion.div>
      </div>
    );
  }

  if (snapshot?.status === "error") {
    return (
      <div className="boot">
        <div className="boot-card boot-error">
          <p className="overline">Boot failure</p>
          <h2>
            The engine could not <em>start.</em>
          </h2>
          <pre className="boot-cmd mono">{snapshot.boot_error}</pre>
          <p className="dim">Check that EuroSAT is in data/eurosat/2750 and the model weights are cached, then restart the engine.</p>
        </div>
      </div>
    );
  }

  const progress = Math.max(boot.progress, 0.02);
  return (
    <div className="boot">
      <motion.div className="boot-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <div className="boot-orbit spinning" aria-hidden="true">
          <span className="boot-sat" />
        </div>
        <p className="overline">Payload boot sequence</p>
        <h2>
          Bringing the <em>payload</em> online
        </h2>
        <div className="boot-bar" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
          <motion.span animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.6, ease: [0.2, 0.7, 0.1, 1] }} />
        </div>
        <ol className="boot-stages">
          {STAGES.map((s, i) => {
            const next = STAGES[i + 1]?.at ?? 1.01;
            const state = progress >= next ? "done" : progress >= s.at ? "active" : "todo";
            return (
              <li key={s.text} className={`boot-stage ${state}`}>
                <span className="boot-tick" aria-hidden="true" />
                {s.text}
              </li>
            );
          })}
        </ol>
        <p className="dim mono boot-note">{boot.stage || snapshot?.boot_stage || "Waiting for engine…"}</p>
      </motion.div>
    </div>
  );
}
