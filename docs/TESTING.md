# Testing

Run everything with `python -m pytest -q` (~1 min). pytest configuration lives in `pyproject.toml`
(`testpaths = ["tests"]`, `--strict-markers`, user/deprecation warnings ignored). The real-data test skips if `data/`
is missing. Use `HF_HUB_OFFLINE=1` where weights are loaded.

## Principles
* Random tensors are acceptable **only** for unit-test plumbing; accuracy claims use real EuroSAT chips.
* Tests that guard a past bug are named for the invariant they protect (see
  [METHODOLOGY.md §9](METHODOLOGY.md#9-validity-invariants-each-was-a-real-bug)).

## Test map

| File | Covers |
|---|---|
| `test_ieee754.py` | float32 bit inspection, exponent-MSB explosion, sign flip, vectorised flips, random SEUs, MBU clusters |
| `test_fault_injector.py` | Targeted bit flips through `ModelFaultInjector`, record keeping |
| `test_radiation_model.py` | Periodic SAA passes, Poisson step generation, reset |
| `test_orbit_model.py` | Sentinel-2A geometry, SAA over the South Atlantic, SAA dominates upsets, interface parity with the periodic model |
| `test_layer_registry.py` | ResNet-18 block partition, block-selective restore |
| `test_monitors.py` | T1 detection, T2 drift, T3 checksum, orchestrator diagnosis, **BN `running_var` sign flip restores its own block** (upstream T3 check) |
| `test_repair.py` | Golden store round-trip/compression, BN recalibrator, selective restore end-to-end |
| `test_baselines_and_sim.py` | Ranger, BLINK, scrubbing, confusion analyser, harness mini-run, scrub sweep + Pareto points, monitor-compute energy only on monitored batches, CRC audit repairs silent faults, summary reports audit strategy |
| `test_correctness_fixes.py` | SAA overlap integrated over the step (not by endpoint), forward-order primary suspect, restore covers all CRC-confirmed blocks, recalibration energy charged, faults uniform over bits, **pretrained model accuracy on real EuroSAT**, monitor sees raw activations when Ranger attaches after, ground uplink repairs only at contacts, INT8 store smaller + rebaseline clears alarm |
| `test_mobilenet.py` | MobileNetV2 blocks cover every parameter, alarms map to the struck block, `assert_held_out` rejects training chips, held-out set covers campaign splits |
| `test_report.py` | Student-t interval, bandwidth-match/dominance logic, report from real harness runs |
| `test_export_frontend.py` | Frontend export replays the fault stream and writes the index |

## What is *not* tested
* Frontend UI (only its data export); verify with `npm run typecheck` / `npm run build` in `frontend/`.
* Streamlit dashboard rendering.
* The Colab notebook.
* Statistical power of campaign conclusions (3 seeds); that is a methodology limit, not a unit-test one.

## Adding tests
1. Prefer the small fixtures (`small_setup`, `base_test_env`, `repair_setup`, `test_setup`) already defined per file.
2. For any new evaluation-validity rule add a regression test to `test_correctness_fixes.py`.
3. For a new architecture add the three `test_mobilenet.py`-style checks: block coverage, alarm→block mapping,
   held-out enforcement.
4. Keep real-data tests skippable when `data/` is absent.
