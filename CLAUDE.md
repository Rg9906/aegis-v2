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
- Dashboard: `streamlit run dashboard/app.py`

## Layout
`aegis/fault_engine` (IEEE-754 flips, LEO/SAA flux model, injector) · `aegis/monitor` (T1 activation
envelope, T2 BN drift, T3 CRC-32, orchestrator) · `aegis/repair` (zstd golden store, restorer, BN
recalibration) · `aegis/baselines` · `aegis/simulation` (mission harness, confusion analysis) ·
`aegis/data` (real EuroSAT splits) · `aegis/reporting.py` (shared loaders/colors for figures + dashboard).

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
- SAA flux is integrated over the minutes of each step that fall inside a pass window.

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
- Not done: second architecture (no EuroSAT MobileNet checkpoint; needs fine-tuning), real orbit
  /AP8 SAA map (periodic pass model only).
- Known limitation: T1/T2 hook compute is not in the energy model (CRC scans are);
  the checkpoint's training split is unknown, so clean accuracy may be optimistic (relative
  degradation is unaffected).
