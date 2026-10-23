# Frontend: Briefing + Live Console

`frontend/` is one React 19 + TypeScript + Vite app with two modes, switched by the pill in the header
(`src/Root.tsx`, hash routing):

* **Briefing** (default, `#overview`): the scrolling "mission dossier" over recorded campaigns. No backend; it
  reads pre-exported JSON (sections 1-6 below).
* **Live console** (`#/console/lab`, `#/console/mission`, `#/console/inspect`): a control room driving the real
  engine (`aegis/console/`) over REST + one WebSocket (section 7).

The earlier Streamlit console was removed.

## 0. Live console

Start the engine from the repository root (venv active), then open the console:

```powershell
python -m aegis.console            # API + built UI on http://127.0.0.1:8765/#/console  (--open, --arch mobilenet_v2)
cd frontend; npm run dev           # optional: UI dev server on :5173, proxies /api to :8765
```

| Workspace | What runs on the engine |
|---|---|
| Strike Lab | Flip chosen bits (block, IEEE-754 field, 1/4/16 bits) in the live payload; measure accuracy on a 30-chip probe; with the autopilot armed, one monitored batch -> T1/T2 suspects -> T3 CRC confirmation -> selective restore -> re-measure. Manual health check, Aegis repair, CRC audit, full scrub, reset. |
| Live Mission | 4 replicas (unprotected, periodic scrub, Aegis-v2, Aegis+audit) fly the real Sentinel-2A track with the identical seeded fault stream; every step streams live (map, strategy lanes, accuracy chart, damaging-hit feed, debrief). Radiation 1x = Alsat-1 calibrated; 10x/30x are labelled accelerated. |
| Inspector | Classify any of the 150 held-out chips with the live payload vs the golden model; per-class accuracy and confusion matrix on the probe or all chips. |

The UI replays engine events as a sequence with a minimum dwell per stage (`lab/Lab.tsx`), so a fast engine still
reads step by step; every number shown is computed by the engine. The flight log streams every engine action.
Keys: `1/2/3` workspaces, `F` fire, `L` log.

## 1. Data pipeline

```
results/<campaign>/seed_*.json ──► experiments/export_frontend_data.py ──► frontend/public/data/<campaign>.json
                                                                           frontend/public/data/index.json
```

`python experiments/export_frontend_data.py --name <campaign> [--results-dir results] [--out-dir frontend/public/data]`

The exporter does more than copy:
1. **Replays each seed's fault stream** (same seed, orbit, architecture, fault field) to recover *where* every flip
   landed (block, tensor, from/to value, bit).
2. **Checks the replay against the recorded `bit_flips_step`** of every step and aborts if they disagree. This guards
   against exporting data from a different code version.
3. Adds statistics (rows, Pareto points, bandwidth verdicts, whether hook-compute energy was added analytically),
   the real ground track and a geomagnetic-field grid for the globe (TLE orbits only), block sizes and key constants.
4. Rewrites `index.json` with every campaign JSON present in the folder.

Exported now: `main_final`, `main_final_mobilenet` and the pre-fix `main_gpu`.

## 2. Running it

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
npm run typecheck    # tsc -b
npm run build        # tsc -b && vite build -> frontend/dist (static, base "./", works from any folder/host)
npm run preview
```

The default campaign is `main_final`, else any `main_final*`, else any `main*`, else the first. Select another with the
nav dropdown or `?campaign=<name>` in the URL.

## 3. Source layout

| Path | Role |
|---|---|
| `src/main.tsx`, `src/App.tsx` | Entry; loads `data/index.json`, then the chosen campaign; composes sections |
| `src/lib/data.ts` | `loadIndex`, `loadCampaign`, `isPreFix`, `accuracySeries`, fault `severity` classification |
| `src/lib/types.ts` | `Campaign`, `TelemetryStep`, `FaultHit` schemas (must match the exporter) |
| `src/lib/geo.ts`, `format.ts`, `useMissionClock.ts` | Projection helpers, number formatting, replay clock + theme hook |
| `src/components/Hero.tsx`, `Globe.tsx` | Headline numbers and the orbit/SAA globe (d3-geo, d3-contour, world-atlas) |
| `src/components/HowItWorks.tsx`, `Method.tsx` | Explanation of the pipeline and the methodology/assumptions |
| `src/components/replay/*` | Mission replay: `Replay`, `Timeline`, `AccuracyChart`, `MemoryMap`, `GroundMap`, `Telemetry` |
| `src/components/Pareto.tsx`, `Scoreboard.tsx` | Accuracy-vs-bandwidth chart and the strategy table |
| `src/components/Incidents.tsx` | Narrative of incidents INC-01/02 (shown as pre-fix for old campaigns) |
| `src/styles/tokens.css`, `app.css` | Design tokens (light/dark) and layout |

Stack (pinned): react 19.3.1, vite 8.3.5, typescript 5.9.3, d3-array/contour/geo/scale/shape, topojson-client,
world-atlas, motion, Geist and Instrument Serif fonts via `@fontsource`.

## 4. Pre-fix handling

`isPreFix(campaign)` is true when no `acc_aegis_audit` strategy exists. Such campaigns (e.g. `main_gpu`) predate the
upstream-localisation fix and the CRC audit; the Incidents section labels them accordingly. Do not present pre-fix numbers
as final.

## 5. Deployment

`npm run build` produces `frontend/dist/` with relative asset paths, so it can be served from any static host or opened
from a sub-folder. `dist/` and `node_modules/` are gitignored. Re-export data before building if campaigns change.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| "No mission data" screen | Run the export script; confirm `frontend/public/data/index.json` exists |
| Exporter exits "Fault replay ... does not match recorded flip counts" | Campaign was produced by different code/orbit/arch; re-run the campaign or check out the matching commit |
| Blank globe | Campaign used `--orbit periodic` (no ground track is exported) |
| Type errors after changing the exporter | Update `src/lib/types.ts` to match the payload |
