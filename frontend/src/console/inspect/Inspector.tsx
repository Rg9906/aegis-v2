import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { blockShort } from "../../lib/format";
import { api, chipUrl, useEngine } from "../engine";
import { CLASSES, CLASS_SHORT } from "../constants";
import { pct1 } from "../fx";
import type { ClassifyResult, Evaluation, Gallery, TopK } from "../types";

/** Real EuroSAT gallery: classify any chip with the live payload and the golden model side by side. */
export default function Inspector() {
  const { snapshot, run, busy } = useEngine();
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [cls, setCls] = useState<number | "all">("all");
  const [sel, setSel] = useState<number | null>(null);
  const [result, setResult] = useState<ClassifyResult["results"][number] | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const disabled = !!busy || !!snapshot?.mission_running;

  useEffect(() => {
    api
      .get<Gallery>("gallery")
      .then((g) => {
        setGallery(g);
        setSel((s) => s ?? g.watch[0] ?? 0);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [snapshot?.arch]);

  // Classify the latest selection (also after the payload changes). Requests are serialized and
  // a selection that changed mid-flight is re-run, so fast clicking never races the engine.
  const want = useRef<number | null>(null);
  const inflight = useRef(false);
  const [classifying, setClassifying] = useState(false);
  const classifyLatest = useCallback(async () => {
    if (inflight.current || want.current === null) return;
    inflight.current = true;
    setClassifying(true);
    const target = want.current;
    try {
      const r = await api.post<ClassifyResult>("classify", { indices: [target] });
      if (want.current === target) {
        setResult(r.results[0]);
        setError(null);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      inflight.current = false;
      setClassifying(false);
      if (want.current !== target) void classifyLatest();
    }
  }, []);
  useEffect(() => {
    if (sel === null || snapshot?.mission_running) return;
    want.current = sel;
    void classifyLatest();
  }, [sel, snapshot?.strikes, snapshot?.repairs, snapshot?.mission_running, classifyLatest]);

  const chips = useMemo(() => (gallery ? gallery.chips.filter((c) => cls === "all" || c.label === cls) : []), [gallery, cls]);
  const corrupted = snapshot?.corrupted ?? [];

  const evaluate = (scope: "probe" | "full") =>
    run<Evaluation>(scope === "full" ? "Evaluate gallery" : "Evaluate probe", "evaluate", { scope }).then((r) => r && setEvaluation(r));

  if (!gallery && error) return <p className="m-error">Gallery unavailable: {error}</p>;

  return (
    <div className="inspect">
      <section className="panel-glass insp-gallery" aria-label="EuroSAT gallery">
        <div className="block-head">
          <h3>Sentinel-2 chips · held-out evaluation split</h3>
          <span className="hint">{gallery?.chips.length ?? 0} chips · 64 px RGB, 10 m/px</span>
        </div>
        <div className="chips insp-filter" role="group" aria-label="Filter by class">
          <button className={`chip ${cls === "all" ? "" : "chip-off"}`} onClick={() => setCls("all")}>
            All
          </button>
          {CLASSES.map((c, i) => (
            <button key={c} className={`chip ${cls === i ? "" : "chip-off"}`} onClick={() => setCls(i)}>
              {CLASS_SHORT[i]}
            </button>
          ))}
        </div>
        <div className="insp-grid">
          {chips.map((c) => (
            <button key={c.index} className={`insp-chip ${sel === c.index ? "on" : ""}`} onClick={() => setSel(c.index)} aria-label={`Chip ${c.index}, ${CLASSES[c.label]}`} aria-pressed={sel === c.index}>
              <img src={chipUrl(c.index)} alt="" width={64} height={64} loading="lazy" />
            </button>
          ))}
        </div>
      </section>

      <section className="insp-side">
        <div className={`panel-glass insp-detail ${classifying ? "busy" : ""}`}>
          {snapshot?.mission_running && <p className="insp-warn">A live mission is running; classification resumes when it ends.</p>}
          {error && gallery && <p className="insp-warn">{error}</p>}
          {corrupted.length > 0 && (
            <p className="insp-warn">
              Payload carries corrupted blocks ({corrupted.map(blockShort).join(", ")}). Predictions below come from the damaged model.{" "}
              <a href="#/console/lab">Repair in the Strike Lab</a>
            </p>
          )}
          {sel !== null && (
            <div className="insp-hero">
              <motion.img key={sel} src={chipUrl(sel, 4)} alt="Selected chip" width={256} height={256} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} />
              <div>
                <p className="overline">Ground truth</p>
                <h2>{result ? CLASSES[result.label] : "…"}</h2>
                <p className="dim mono">{result?.path}</p>
                {result && (
                  <p className={`insp-verdict ${result.live.pred === result.label ? "ok" : "bad"}`}>
                    {result.live.pred === result.label ? "Payload correct" : !result.live.finite ? "Payload output is NaN" : `Payload says ${CLASSES[result.live.pred ?? 0]}`}
                  </p>
                )}
              </div>
            </div>
          )}
          <div className="insp-compare">
            <Bars title="Live payload" top={result?.live.top} label={result?.label} tone={corrupted.length ? "warn" : "live"} />
            <Bars title="Golden model" top={result?.golden.top} label={result?.label} tone="golden" />
          </div>
        </div>

        <div className="panel-glass">
          <div className="block-head">
            <h3>Accuracy audit</h3>
            <div className="seg seg-sm">
              <button onClick={() => evaluate("probe")} disabled={disabled}>
                Probe · {snapshot?.probe_size}
              </button>
              <button onClick={() => evaluate("full")} disabled={disabled}>
                All {snapshot?.gallery_size}
              </button>
            </div>
          </div>
          <AnimatePresence mode="wait">
            {evaluation ? (
              <motion.div key={`${evaluation.scope}-${evaluation.ms}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <p className="insp-acc">
                  <strong className="mono">{pct1(evaluation.live)}</strong> <span className="dim">payload</span> ·{" "}
                  <strong className="mono">{pct1(evaluation.golden)}</strong> <span className="dim">golden · {evaluation.n} chips in {(evaluation.ms / 1000).toFixed(1)} s</span>
                </p>
                <div className="pc-bars">
                  {evaluation.per_class.map((p, i) => (
                    <div key={p.class} className="pc-row">
                      <span className="pc-name">{CLASS_SHORT[i]}</span>
                      <span className="pc-track">
                        <motion.span className="pc-golden" initial={{ width: 0 }} animate={{ width: `${(p.golden ?? 0) * 100}%` }} transition={{ duration: 0.6, delay: i * 0.03 }} />
                        <motion.span className="pc-live" initial={{ width: 0 }} animate={{ width: `${(p.live ?? 0) * 100}%` }} transition={{ duration: 0.6, delay: 0.1 + i * 0.03 }} />
                      </span>
                      <span className="mono pc-val">{pct1(p.live)}</span>
                    </div>
                  ))}
                </div>
                <Confusion m={evaluation.confusion} />
              </motion.div>
            ) : (
              <p className="dim">Measure the live payload against the golden model per class, with a confusion matrix.</p>
            )}
          </AnimatePresence>
        </div>
      </section>
    </div>
  );
}

function Bars({ title, top, label, tone }: { title: string; top?: TopK[]; label?: number; tone: string }) {
  return (
    <div className={`bars bars-${tone}`}>
      <p className="overline">{title}</p>
      {(top ?? []).map((t, i) => {
        const p = typeof t.p === "number" ? t.p : 0;
        return (
          <div key={`${t.cls}-${i}`} className={`bar-row ${t.cls === label ? "truth" : ""}`}>
            <span className="bar-name">{CLASS_SHORT[t.cls]}</span>
            <span className="bar-track">
              <motion.span initial={{ width: 0 }} animate={{ width: `${p * 100}%` }} transition={{ duration: 0.55, delay: i * 0.06 }} />
            </span>
            <span className="mono bar-val">{typeof t.p === "number" ? `${(p * 100).toFixed(1)}%` : t.p}</span>
          </div>
        );
      })}
      {!top && <p className="dim">Select a chip</p>}
    </div>
  );
}

function Confusion({ m }: { m: number[][] }) {
  const max = Math.max(1, ...m.flat());
  return (
    <div className="confusion" role="table" aria-label="Confusion matrix, rows true class, columns predicted">
      <div className="cm-grid">
        {m.map((row, i) =>
          row.map((v, j) => (
            <span
              key={`${i}-${j}`}
              className={`cm-cell ${i === j ? "diag" : v ? "off" : ""}`}
              style={{ ["--v" as string]: v / max }}
              title={`true ${CLASSES[i]} → predicted ${CLASSES[j]}: ${v}`}
            >
              {v || ""}
            </span>
          )),
        )}
      </div>
      <p className="dim cm-caption">Rows: true class · columns: predicted (checkpoint order). Off-diagonal = errors.</p>
    </div>
  );
}
