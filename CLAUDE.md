# Project Aegis-v2

Research prototype: detection-gated, layer-selective restore of a radiation-corrupted onboard
EO classifier (ResNet-18 on EuroSAT) vs. Ranger clipping, BLINK re-centering, periodic scrubbing.
Design rationale: `AEGIS_RESEARCH_STUDY_AND_PROTOTYPE_BLUEPRINT.md`; prior-art map in the
`Project Aegis_ Prior-Art Map ... .md` file.

## Environment
- Use the Python 3.12 that has the deps:
  `C:\Users\RG Saran Vishakan\AppData\Local\Microsoft\WindowsApps\python3.12.exe`.
  The default `python` on PATH is an unrelated venv (no torch).
- Python `requests` fails SSL to huggingface.co on this machine; `curl` works. Weights
  (`cm93/resnet18-eurosat`) are in the HF cache and load with `local_files_only`; set
  `HF_HUB_OFFLINE=1` for scripts.
- EuroSAT RGB lives in `data/eurosat/2750/` (gitignored). Re-fetch:
  `curl -L -o data/eurosat/EuroSAT.zip https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip`.

## Commands
- Tests: `python -m pytest -q` (~1 min; the real-data test skips if `data/` is missing)
- Demo: `python demo.py`
- Campaign: `python experiments/run_campaign.py --days 3 --seeds 0 1 2 --eval-per-class 15 --scrub-sweep-hours 1 3 6 24 --name main`
  (~35 s/step on CPU with 9 replicas; writes `results/<name>/seed_*.json`, `campaign.json`)
- Figures: `python experiments/make_figures.py --name main` -> `results/<name>/figures/`
- Report: `python experiments/write_report.py --name main` -> `results/<name>/report.md`
  (seed 95% t-intervals, paired Aegis-minus-baseline deltas, scrub-Pareto verdict)
- Second architecture: `python experiments/train_mobilenet.py --epochs 3 --device cuda` (GPU; writes
  `checkpoints/`, gitignored), then `run_campaign.py ... --arch mobilenet_v2`
- GPU runs: `notebooks/aegis_colab.ipynb` (repo `Rg9906/aegis-v2`, token in Colab secret `GH_TOKEN`)
- Dashboard: `streamlit run dashboard/app.py`
- Frontend (React mission dossier, replaces the Streamlit dashboard for presentation):
  `python experiments/export_frontend_data.py --name <campaign>` (writes `frontend/public/data/`,
  replays each seed's fault stream and checks it against recorded flip counts), then in `frontend/`:
  `npm install`, `npm run dev` (http://localhost:5173) or `npm run build` (static site in `dist/`).
  Campaigns without the audit strategy are shown as "pre-fix" (INC-01/02) automatically.
  Sections: hero/globe, animated pipeline, IEEE-754 bit-flip anatomy (real recorded hits), replay, Pareto,
  costs (bandwidth vs energy, honest), scoreboard, two-architecture comparison (needs both `main_final*` exports),
  incidents, method. New-section styles live in `frontend/src/styles/sections.css`.
- Paper: `researchpaper/` (IEEEtran two-column). `python researchpaper/make_paper_figures.py` regenerates
  `figures/*.pdf`; build with `pdflatex -> bibtex -> pdflatex x2` in `researchpaper/`. Entries marked `% VERIFY` in
  `references.bib` are unverified 2026 preprints from the prior-art map. Paper numbers come from the final campaigns.

## Layout
`aegis/fault_engine` (IEEE-754 flips, LEO/SAA flux model, injector) · `aegis/monitor` (T1 activation
envelope, T2 BN drift, T3 CRC-32, orchestrator) · `aegis/repair` (zstd golden store, restorer, BN
recalibration) · `aegis/baselines` · `aegis/simulation` (mission harness, confusion analysis) ·
`aegis/data` (real EuroSAT splits) · `aegis/models/factory.py` (`--arch` backbones) ·
`aegis/reporting.py` (shared loaders/colors for figures + dashboard).

## Non-obvious invariants (each was a real bug)
- **Model input**: checkpoint expects 224 px bicubic-upsampled chips with EuroSAT mean/std, and
  logits in the checkpoint's `label_names` order (`EUROSAT_CLASSES`), not alphabetical.
  64 px input gives ~18% accuracy; alphabetical labels give chance.
- **Never evaluate on `torch.randn`**: random tensors are fine for unit-test plumbing only.
- **T1 bounds = observed clean min/max + margin** over >=100 calibration chips. A mean +/- k*sigma
  envelope fires on every clean batch (heavy-tailed activations), which silently turns the
  "detection gate" into restore-everything.
- **Calibration and evaluation chips must be disjoint** (`load_eurosat_splits`).
- **Do not call `refresh_golden_checksums()` while any block may be corrupt**: it bakes the
  corruption into the T3 reference. With the lossless golden store, restore is bit-exact and
  needs no BN recalibration (recalibration is off by default and is charged inference energy).
- Faults are placed uniformly over stored bits (block and tensor chosen proportional to size);
  default `fault_field="any"`. Forcing `"exponent"` inflates severity: disclose it if used.
- Alarm-to-block mapping uses `LayerRegistry.classify_key` for both parameter keys and module
  names (ResNet `layerX.Y`, MobileNetV2 `features.N`). A separate mapping with a "stem" fallback
  once sent every MobileNetV2 alarm to the wrong block.
- MobileNetV2 is trained only on chips outside the campaign splits; its checkpoint sidecar lists
  the held-out chips and `run_campaign.py` refuses a split that is not inside them.
- Monitoring energy charges T1/T2 hook compute only on the monitored incoming batch (evaluation
  passes go through the same hooks but are measurement, not mission work).
- T3 also CRC-checks the block upstream of the earliest T1/T2 alarm: T1 hooks conv outputs and T2
  hooks BN inputs, so a BN-parameter fault (seed 1: sign flip on `layer3.0.bn2.running_var` -> NaN)
  first shows one block downstream and the culprit was never checked (accuracy stuck at 10%).
- Behavioral detection cannot see sub-threshold faults (seed 1, step 49: a weight 0.106 -> 1.7 cost
  17 points with no alarm). The "Aegis-v2 + CRC audit" strategy (`--audit-hours`, default 6)
  CRC-scans all blocks periodically and restores mismatches; report it alongside plain Aegis-v2.
- Orbit: default `--orbit tle` = real Sentinel-2A track (SGP4 on `orbits/sentinel2a.tle`, CelesTrak)
  with the SAA defined physically as low |B| from IGRF (rate ramps 20,000 -> 18,000 nT). This
  yields ~80 SAA-core min/day and ~82% of upsets in the SAA.
  `--orbit periodic` keeps the old synthetic fixed passes. Either way flux is integrated per minute.
- Upset rates are calibrated (2026-10-21) to the measured Alsat-1 mean, 4.04e-7 upsets/bit/day with
  98.55% single-bit (Bentoutou 2011, full text verified; source in `RadiationEnvironmentConfig`).
  Still assumed: the SAA-core/background ratio (75.3x) and transfer from Alsat-1 (686 km, 2002 SRAM)
  to Sentinel-2A (786 km, modern memory). The Flying Laptop 73.41% SAA figure is unverified.

## Status (2026-10-21)
- Phases 1-4 implemented; evaluation-validity fixes above applied and covered by
  `tests/test_correctness_fixes.py`.
- Results before 2026-10-21 (e.g. "98.9% energy savings") are invalid: random-noise inputs,
  always-on gate, uncharged recalibration. Do not cite them.
- Day 5: dashboard + figures implemented; headline numbers come from `results/main/`.
- Added 2026-10-21: Ranger+Aegis combo (monitor hooks registered before clamp hooks so T1 sees
  raw activations), ground-uplink baseline (repairs only in contact windows, 1 Mbps, 60 MB/pass),
  T3 CRC scan energy (assumed 20 pJ/bit DRAM read), optional INT8 golden store (`--golden-int8`;
  41.6 -> 9.2 MB, -0.7 pts on 1,000 chips; restored blocks are re-baselined individually),
  transient activation-fault experiment (`experiments/transient_faults.py`: 0% spurious restores).
- Not done: AP8/AE9
  trapped-flux model + device cross-section (IGRF |B| ramp is the proxy).
- Added 2026-10-21 (later): T1/T2 hook compute energy (4 passes per Conv/Linear output + 3 per BN
  input at an assumed 6.9 pJ/op = 12.5 mJ / 1.82 G MACs): ~1.9 mJ per 16-image ResNet-18 batch,
  ~138 mJ over 3 days, i.e. ~10 full-reload equivalents of memory energy, so it matters for the
  energy comparison. Report generator adds it analytically to campaigns run before it existed.
  MobileNetV2 path trained and run on Colab (see final campaigns below).
- Final campaigns (2026-10-22, Colab T4, 3 seeds x 3 days, real orbit, audit strategy included):
  `results/main_final/` (ResNet-18, clean 89.3%) and `results/main_final_mobilenet/` (MobileNetV2,
  clean 99.3%, held-out split verified; checkpoint sidecar in its `checkpoints/`). Headline: Aegis-v2
  holds 87.4% (ResNet) / 99.3% (MobileNetV2) mean accuracy at 188 / 3.4 MB/day; scrubbing needs
  889 / 202 MB/day to match. Aegis does NOT save energy: monitor hook compute makes it ~125-144
  mJ/day vs ~6-28 mJ/day for 12 h scrubbing. Claim is bandwidth, not energy. 3 seeds and 150 eval
  chips give wide intervals; MobileNetV2 seed 2 had no damaging fault. `results/main_gpu/` is
  pre-fix; don't cite. Frontend data exported for both final campaigns.
- Fixed 2026-10-22: `preprocess_chips` built mean/std on CPU, crashing GPU MobileNetV2 training.
- Known limitations: Ranger/BLINK per-inference compute is not charged; the ResNet-18 checkpoint's
  training split is unknown, so its clean accuracy may be optimistic (relative degradation is
  unaffected; the MobileNetV2 path has a verified held-out split).
