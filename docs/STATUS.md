# Project Status and History

Dated record of what exists, what was fixed, and what is invalid. For the short operating version see `CLAUDE.md`.
Last updated: 2026-10-22.

## Current state
* Phases 1-4 implemented (fault engine, monitors, repair, baselines, simulation, reporting).
* Evaluation-validity fixes applied and covered by `tests/test_correctness_fixes.py`.
* Final campaigns complete (2026-10-22, Colab T4, 3 seeds x 3 days, real orbit, audit included):
  `results/main_final/` (ResNet-18) and `results/main_final_mobilenet/` (MobileNetV2). Frontend data exported for both.
* **Headline:** Aegis-v2 holds 87.4 % (ResNet-18) / 99.3 % (MobileNetV2) mean accuracy at 188 / 3.4 MB/day; scrubbing
  needs 889 / 202 MB/day to match. **It does not save energy** (about 125-144 mJ/day vs about 6-28 mJ/day for 12 h scrubbing). The claim
  is bandwidth.

## Timeline

| Date | Change |
|---|---|
| before 2026-10-21 | First prototype. Results (e.g. "98.9 % energy savings") **invalid**: random-noise inputs, always-on gate, uncharged recalibration |
| 2026-10-21 | Real EuroSAT evaluation; T1 bounds from observed min/max; disjoint calibration/evaluation splits; lossless golden store, recalibration off by default; correct alarm-to-block mapping; fix for `refresh_golden_checksums` misuse |
| 2026-10-21 | Real Sentinel-2A orbit (SGP4 + IGRF-defined SAA); rates calibrated to Alsat-1 measured mean (4.04e-7 upsets/bit/day, 98.55 % single-bit) |
| 2026-10-21 | Added Ranger+Aegis combo, ground-uplink baseline, T3 CRC scan energy (20 pJ/bit), optional INT8 golden store, transient-fault experiment (0 % spurious restores) |
| 2026-10-21 | T1/T2 hook compute energy charged (6.9 pJ/op); statistical report (t-intervals, paired deltas, Pareto verdict); MobileNetV2 path; upstream T3 check for BN-parameter faults |
| 2026-10-21 | CRC audit strategy for silent sub-threshold faults; React mission dossier frontend |
| 2026-10-22 | Final ResNet-18 and MobileNetV2 campaigns on Colab; fix: `preprocess_chips` mean/std now on the correct device (GPU MobileNetV2 training crashed); documentation pass (README + `docs/`) |

## Incidents that shaped the design
* **INC-01** - A BN-parameter fault (seed 1: sign flip on `layer3.0.bn2.running_var` -> NaN) first showed one block downstream;
  the culprit was never CRC-checked and accuracy stayed at 10 %. Fix: T3 also checks the block upstream of the earliest alarm.
* **INC-02** - A sub-threshold fault (seed 1, step 49: a weight 0.106 -> 1.7) cost 17 points with no alarm. Mitigation: periodic
  full CRC audit (`--audit-hours`, default 6).
* **Envelope bug** - mean +/- k sigma T1 bounds fired on every clean batch (restore-everything). Fix: observed min/max + margin.
* **Label/size bug** - 64 px or alphabetical labels give ~18 % / chance accuracy. Fix: 224 px, checkpoint label order.
* **MobileNetV2 mapping bug** - a separate name-to-block mapping with a "stem" fallback mis-routed alarms. Fix: single
  `classify_key`.

## Open items
See the roadmap in [DEVELOPMENT.md](DEVELOPMENT.md#5-roadmap--open-items): AP8/AE9 modelling, sensitivity sweeps, more
seeds, reducing monitor energy, charging Ranger/BLINK compute.

## Documentation index
README.md, docs/ARCHITECTURE.md, METHODOLOGY.md, USER_GUIDE.md, RESULTS.md, TESTING.md, DEVELOPMENT.md, FRONTEND.md,
STATUS.md, GLOSSARY.md.
