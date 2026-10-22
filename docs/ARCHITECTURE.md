# Architecture

This document describes how the Aegis-v2 code base is organised, what each module is responsible for, and how
data flows through one simulated mission step. For *why* the numbers are what they are, see
[METHODOLOGY.md](METHODOLOGY.md).

## 1. Package map

```
aegis/
├── config.py                  Physical + budget constants (dataclasses), EuroSAT class order, preprocessing constants
├── reporting.py               Shared loaders/colours/Pareto helpers used by figures, dashboard, report
├── data/eurosat.py            Real EuroSAT loading, disjoint calibration/evaluation splits, preprocessing
├── models/
│   ├── eurosat_classifier.py  EuroSATResNet18 wrapper (HF checkpoint, predict/evaluate/clone_clean_state)
│   ├── factory.py             --arch backbones (resnet18, mobilenet_v2), held-out-split assertion
│   └── layer_registry.py      Partition of a model into restorable "blocks"; key -> block classification
├── fault_engine/
│   ├── ieee754.py             float32 bit inspection/flipping, SEU and MBU-cluster injection
│   ├── radiation_model.py     LEORadiationEnvironment: synthetic periodic SAA passes, Poisson upsets
│   ├── orbit_model.py         OrbitTrackRadiationEnvironment: SGP4 + IGRF |B| derived SAA (default)
│   └── injector.py            ModelFaultInjector: places flips in a model, records and replays them
├── monitor/
│   ├── activation_envelope.py T1: activation range envelope (NaN/Inf/out-of-range)
│   ├── bn_drift.py            T2: BatchNorm batch-vs-running statistic drift
│   ├── checksum.py            T3: CRC-32 over block parameter bytes
│   └── orchestrator.py        AegisHealthMonitor + HealthReport: T1/T2 -> block mapping -> T3 confirmation
├── repair/
│   ├── golden_store.py        CompressedGoldenStore: zstd+safetensors per block, optional INT8
│   ├── selective_restore.py   DetectionGatedRestorer: HealthReport -> restore -> RepairEvent
│   └── recalibrate.py         BNRecalibrator: optional label-free BN statistic refresh
├── baselines/                 ranger_clipper, blink_recentering, periodic_scrubbing, ground_uplink
└── simulation/
    ├── mission_sim.py         MissionSimulationHarness: all strategies side by side, telemetry + summary
    └── confusion_analyzer.py  EOConfusionAnalyzer: which EO classes get confused under faults
```

Entry points live outside the package: `demo.py`, `experiments/*.py`, `dashboard/app.py`, `frontend/`.

## 2. Core abstractions

### 2.1 `LayerRegistry` (`models/layer_registry.py`)
Partitions a model's `state_dict` into **blocks**, the unit of detection, restoration and energy accounting.
`LayerRegistry.classify_key(key)` is the *single* source of truth for naming:

| Architecture | Key / module prefix | Block id |
|---|---|---|
| ResNet-18 | `conv1`, `bn1` | `stem` |
| ResNet-18 | `layerX.Y.*` | `layerX_blockY` |
| ResNet-18 | `fc` / `classifier` | `classifier` |
| MobileNetV2 | `features.N.*` | `features_blockN` (N = 0 is the stem conv) |
| anything else | – | `other` |

Block order (`block_ids`) is forward-pass order and is used to pick the *most upstream* suspect. Each block carries
`LayerBlockMetadata` (parameter count, fp32/int8 byte sizes, total bits). `get_block_state_dict` /
`restore_block_state_dict` are the read/write primitives used by the golden store and restorer.

> The monitor maps module *names* through the same `classify_key`. A separate mapping once sent every MobileNetV2
> alarm to the wrong block. Never fork it.

### 2.2 Fault engine
* `ieee754.py` – bit-level float32 manipulation: `inspect_float32_bits`, `flip_bits_in_tensor`,
  `inject_single_bit_flip`, `sample_bit_positions` (field-restricted sampling), `inject_random_seus`,
  `inject_mbu_cluster`. Constants `FLOAT32_EXPONENT_MSB` (bit 30) and `TargetFieldType` (`any|exponent|mantissa|sign`).
* `radiation_model.LEORadiationEnvironment` – synthetic fixed SAA passes (`--orbit periodic`).
* `orbit_model.OrbitTrackRadiationEnvironment` – real Sentinel-2A track (`--orbit tle`, default). Same interface
  (`step(delta_minutes, total_model_bits) -> OrbitalState`, `elapsed_minutes`, `elapsed_days`, `current_orbit_index`).
* `injector.ModelFaultInjector` – turns an `OrbitalState` into bit flips on a model. Only the *reference*
  (unmitigated) injector samples randomly; every other replica **replays** the exact
  `(block_id, param_name, flat_index, bit_position)` records via `inject_targeted_bit_flip`. This is what makes the
  comparison paired and fair. Faults are placed uniformly over stored bits (block, then tensor, proportional to size).
  `rollback_all_faults()` undoes them (tests/demo).

### 2.3 Monitors
| Tier | Class | Hooks | Fires when | Cost counter |
|---|---|---|---|---|
| T1 | `ActivationEnvelopeMonitor(margin=0.5)` | Conv/Linear outputs | NaN/Inf, or value outside clean `[min,max]` widened by `margin × range` | `ops_performed` |
| T2 | `BNDriftMonitor(drift_threshold=3.5)` | BN inputs | mean of (|Δμ|/σ_run + |Δσ|/σ_run) over channels exceeds calibrated tolerance / threshold; NaN → 9999 | `ops_performed` |
| T3 | `ChecksumMonitor(registry)` | none (on demand) | CRC-32 of block parameter bytes differs from golden | `bytes_scanned` |

`AegisHealthMonitor` (orchestrator) owns all three:

1. `calibrate(calib_images)` – profile T1 bounds, T2 tolerances, T3 golden CRCs.
2. `attach()` / `detach()` – register / remove T1+T2 forward hooks.
3. `clear()` then a forward pass on an incoming batch fills `latest_violations` / `latest_drifts`.
4. `diagnose_current_state(run_tier3_scan=False)` → `HealthReport`:
   * map flagged layer names to blocks (`_map_layer_name_to_block_id`);
   * **add the block just upstream of the earliest candidate** (T1 hooks conv outputs and T2 hooks BN inputs, so a
     BN-parameter fault first shows one block downstream);
   * **confirm each candidate with T3**. Healthy iff nothing is confirmed. Blocks are sorted in forward order and the
     most upstream one is the `primary_suspect_block`.
5. `refresh_golden_checksums()` (dangerous, see below) and `rebaseline_blocks(ids)` (safe, per restored block).

`HealthReport` fields: `is_healthy`, `flagged_blocks`, `primary_suspect_block`, `tier1_violations`,
`tier2_drifts`, `tier3_corrupted_blocks`, and the property `has_catastrophic_anomaly` (NaN/Inf or |max| > 1e10).

### 2.4 Repair
* `CompressedGoldenStore` – simulates rad-hard NVM (SPI NOR / MRAM). One zstd-compressed safetensors blob per block;
  `retrieve_block` decompresses and accounts bytes read and energy (flash read + DRAM write). `quantize_int8=True`
  stores weights of dim ≥ 2 as per-output-channel symmetric INT8 with fp32 scales (≈ 41.6 → 9.2 MB) – restores
  become lossy and the caller must `rebaseline_blocks`.
* `DetectionGatedRestorer.process_health_report(report, auto_recalibrate)` – returns `None` if healthy, otherwise
  restores `tier3_corrupted_blocks` (forward order); falls back to the primary suspect if T3 information is absent.
  Emits a `RepairEvent` (blocks, bytes, energy split into memory vs recalibration, downtime ms).
  `restore_full_model()` is the blind-scrub primitive.
* `BNRecalibrator` – optional label-free BN statistic refresh from a small unlabeled buffer. **Off by default**:
  with a lossless store restores are bit-exact, recalibration mutates BN running stats in *every* block (invalidating
  T3 references) and is charged one inference per buffered image.

### 2.5 Baselines
| Class | Mechanism | Repairs weights? | Bytes/energy charged |
|---|---|---|---|
| `RangerActivationClipper(margin=0.1)` | clamp activations to calibrated bounds | no | none |
| `BlinkRecenteringMitigator(drift_threshold=3.0)` | re-normalise anomalous BN activations to reference stats | no | none |
| `PeriodicScrubbingController` | `restore_full_model()` every N h | yes (blind) | yes |
| `GroundUplinkReloader` | queue CRC-confirmed blocks; reload only inside contact windows, rate-limited | yes (late) | bytes, energy, link seconds |

Ground uplink contacts are evenly spaced (`ground_passes_per_day`, `ground_pass_duration_seconds` from
`SatellitePayloadBudgetConfig`; 1 Mbps × 480 s ≈ 60 MB per pass).

## 3. The simulation harness

`MissionSimulationHarness` builds **independent deep copies** of the base model, one per strategy, each with its own
`LayerRegistry` and injector:

`unmitigated`, `ranger`, `blink`, `scrub` (+ one per `scrub_sweep_hours`), `aegis`, `combo` (Ranger+Aegis),
`uplink`, `audit` (Aegis+audit).

### Per-step protocol (`run_simulation(total_days, step_minutes=60)`)

```
for each step:
  1. orbital_state = rad_env.step(step_minutes, total_model_bits)        # Poisson upsets, SAA-weighted
  2. records = inj_unmitigated.inject_orbital_flux_step(orbital_state)   # sample once
     replay every record on ranger/blink/scrub/aegis/combo/uplink/audit/sweep replicas
  3. scrub controller (+ sweep controllers) step                         # timer-based reload
  4. incoming = rotating 16-image unlabeled window
     aegis : observe(incoming) -> if unhealthy: restore confirmed blocks
     combo : same, with Ranger clamps attached after the monitor hooks
     audit : same as aegis, plus a full CRC scan whenever an audit epoch boundary passes
     uplink: observe, queue confirmed blocks, reload only during contact windows
  5. evaluate every replica on the labeled evaluation set (chunked)
  6. append TimeStepTelemetry
```

`get_campaign_summary()` condenses the history into mean/final accuracy, energy, bytes, restoration counts, SAA
minutes, golden-store size and the scrub sweep. `TimeStepTelemetry` is the per-step record; its fields are
documented in [RESULTS.md](RESULTS.md).

Design notes:
* **Evaluation vs mission work.** Evaluation passes run through the same hooks but are *measurement*; T1/T2 compute
  energy is charged only on the monitored incoming batch (`_observe`).
* **Hook order.** For Ranger+Aegis the monitor is attached *before* the clamp hooks (forward hooks run in
  registration order) so T1 sees raw, un-clamped activations.
* **Energy of the Aegis family** = golden-store reads + DRAM writes + DRAM reads for T3 CRC scans + T1/T2 hook
  compute (+ recalibration inference if enabled).

## 4. Data and models

* `data/eurosat.py` – `split_eurosat_paths` draws class-stratified, **disjoint** calibration and evaluation chips;
  `load_eurosat_splits` returns an `EuroSATSplits` of preprocessed tensors; `preprocess_chips` does bicubic upsampling
  64 → 224 px and EuroSAT mean/std normalisation; `held_out_paths` / `load_training_chips` support MobileNetV2 training.
* `models/eurosat_classifier.EuroSATResNet18` – loads `cm93/resnet18-eurosat` (`model.safetensors`) offline.
* `models/factory.load_backbone(arch, checkpoint_path)` – resnet18 or mobilenet_v2; `assert_held_out` refuses a
  campaign split that overlaps the MobileNetV2 training data (recorded in the checkpoint sidecar).

## 5. Reporting and front ends

* `aegis/reporting.py` – `STRATEGIES` (summary key, label, colour, telemetry column), `SUMMARY_KEYS`,
  `list_campaigns`, `load_seed_runs`, `pareto_points`, `strategies_present` (older campaigns lack newer strategies).
* `experiments/make_figures.py` → `results/<name>/figures/`.
* `experiments/write_report.py` → `results/<name>/report.md` (95 % Student-t seed intervals, paired deltas, Pareto verdict).
* `dashboard/app.py` – Streamlit viewer over campaign JSON.
* `experiments/export_frontend_data.py` → `frontend/public/data/`; it **replays each seed's fault stream** and checks it
  against the recorded flip counts before exporting. `frontend/` is a React 19 + Vite + TypeScript site (D3 geo/contour
  globe, replay timeline, memory map, Pareto chart). Campaigns without the audit strategy render as "pre-fix" (INC-01/02).

## 6. Dependency direction

```
config ─► data, models, fault_engine, monitor, repair, baselines
models ─► monitor, repair, fault_engine
monitor ─► repair, baselines (ground uplink)
everything ─► simulation ─► experiments / demo / dashboard / frontend export
```
`simulation` is the only layer that knows about every strategy; library modules do not import from `simulation`.
