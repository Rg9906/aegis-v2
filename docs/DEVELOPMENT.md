# Development Guide

## 1. Conventions
* Python ≥ 3.10, type hints, dataclasses for records, docstrings explain *why* (assumptions, sources, pitfalls).
* Constants with physical meaning live in `aegis/config.py` with their source or an "assumed" note.
* Block naming goes **only** through `LayerRegistry.classify_key`.
* Every new number that enters energy/bandwidth must be a named field in `SatellitePayloadBudgetConfig` or
  `RadiationEnvironmentConfig`, with provenance in the comment.
* Keep telemetry/JSON backward compatible; older campaigns must still load (`reporting.strategies_present`).
* `CLAUDE.md` holds the short invariant list; keep it aligned with architecture or status changes, but keep long
  explanations in `docs/`.

## 2. Recipes

### Add a new mitigation baseline
1. Implement it in `aegis/baselines/<name>.py` (attach/detach hooks, or a controller with `step(minutes)`), export it in
   `aegis/baselines/__init__.py`.
2. In `MissionSimulationHarness.__init__` create a deep-copied replica, `LayerRegistry`, and a `ModelFaultInjector`;
   add the injector to the `mirrors` list in `run_simulation` so it receives the **replayed** faults.
3. Evaluate it in the per-step `accs` dict and add `acc_<name>` (and energy/bytes if it repairs) to
   `TimeStepTelemetry`, then to `get_campaign_summary`.
4. Register `(summary_key, label, colour, telemetry_column)` in `aegis/reporting.STRATEGIES`.
5. Extend `write_report.py`/figures if it needs special treatment; add tests in `test_baselines_and_sim.py`.
6. Update the frontend export types if it should appear in the dossier.

### Add a new architecture
1. Add a builder + default checkpoint in `aegis/models/factory.py` and an `--arch` choice in `run_campaign.py`.
2. Extend `LayerRegistry.classify_key` for its module naming; confirm all parameters/buffers are covered
   (`test_mobilenet.py` pattern).
3. Fine-tune only on chips **outside** the campaign splits and write the held-out sidecar; enforce with `assert_held_out`.
4. Check T1 bounds and T2 tolerances calibrate cleanly (no alarms on clean batches).

### Add or tune a detector tier
Implement hook attach/detach, a `clear_*`, an `ops_performed` (or bytes) counter so energy is charged, wire it in
`AegisHealthMonitor.diagnose_current_state`, and verify it does not fire on clean batches before looking at accuracy.

### Add a radiation/orbit model
Match the `step(delta_minutes, total_model_bits) -> OrbitalState` interface plus `elapsed_minutes`, `elapsed_days`,
`current_orbit_index`, `reset()`; register the name in the harness `orbit` switch and `--orbit` choices.

## 3. Pitfalls checklist (read before changing evaluation code)
* Never evaluate on random tensors. Never change input size/normalisation/label order.
* Never call `refresh_golden_checksums()` with possibly corrupt blocks.
* Keep calibration and evaluation splits disjoint; keep ≥ 100 calibration chips.
* Register monitor hooks *before* Ranger clamp hooks.
* Charge monitoring energy on the incoming batch only.
* Do not use recalibration with the lossless store (invalidates T3 references, costs inference energy).

## 4. Known limitations
* No AP8/AE9 trapped-flux model or device cross-section (IGRF |B| ramp proxy).
* SAA/background ratio and Alsat-1→Sentinel-2A transfer are assumptions; solar-cycle variation not modelled.
* ResNet-18 checkpoint training split unknown (clean accuracy may be optimistic).
* Ranger/BLINK per-inference compute not charged.
* Behavioural detectors miss sub-threshold faults (mitigated by the CRC audit, at extra DRAM-read energy).
* Only parameter memory is faulted; activation buffers, control logic and the golden store itself are assumed immune
  (the golden store is assumed rad-hard).
* Small evaluation set and few seeds → wide intervals.
* MobileNetV2: seed 2 had no damaging fault, so that campaign is less informative than ResNet-18.
* Aegis-v2 does not reduce energy (monitor hook compute dominates); only bandwidth is supported.

## 5. Roadmap / open items
Done (2026-10-22): final ResNet-18 and MobileNetV2 campaigns on Colab T4, with the audit strategy and upstream fix.

1. Replace the |B| proxy with AP8/AE9 + device cross-sections (SPENVIS/CREME96).
2. Sensitivity sweeps: fault density, SAA/background ratio, solar cycle.
3. More seeds and a larger evaluation set for tighter intervals (and MobileNetV2 runs with damaging faults in every seed).
4. Reduce monitor hook energy (e.g. monitor a subset of layers or every k-th batch), since it dominates Aegis-v2 energy.
5. Charge Ranger/BLINK per-inference compute for a fully symmetric energy comparison.

## 6. Documentation notes
The `demo.py` module docstring still mentions "5 strategies"; the harness now runs eight (plus the scrub sweep).
`pyproject.toml` uses the blueprint markdown as its `readme`; `README.md` is the entry point for humans. These are
code/config items intentionally left unchanged by the documentation pass.

## 7. Git workflow
Feature branches off `master`; do not commit `data/`, `checkpoints/`, `frontend/node_modules`, `frontend/dist`, or
`results/*.log` (all gitignored). Do not rewrite shared history. Results folders are committed deliberately, with the
code version recorded.
