# Methodology, Assumptions and Validity Rules

This is the scientific contract of the project: what is simulated, what is measured, what is assumed, and which
mistakes invalidate results. Anything marked **ASSUMED** is a modelling choice, not a measurement.

## 1. Research question

Can a *detection-gated, layer-selective* restore keep an onboard classifier near its clean accuracy using far less
memory traffic/energy than blind periodic scrubbing, and without the ground-contact latency of an uplink reload?
Baselines: Ranger clipping, BLINK re-centering, periodic scrubbing, ground uplink, and a Ranger+Aegis combination.

## 2. Workload

* **Task:** EuroSAT RGB land-use classification, 10 classes, 64×64 px Sentinel-2 chips.
* **Model:** ResNet-18 (`cm93/resnet18-eurosat`), clean accuracy ≈ 89.3 % on the campaign evaluation set
  (92.8 % on a 1,000-chip check). A fine-tuned MobileNetV2 is the second architecture.
* **Preprocessing (non-negotiable):** bicubic upsample to **224 px**, normalise with mean
  `(0.3445, 0.3803, 0.4077)` / std `(0.0915, 0.0652, 0.0553)`. 64 px input gives ≈ 18 % accuracy.
* **Label order:** logits follow the checkpoint's `label_names` (`EUROSAT_CLASSES`), *not* alphabetical order
  (alphabetical = chance accuracy).

## 3. Fault model

* **What is corrupted:** the stored float32 parameters and buffers (weights, BN affine/running stats) of the model.
* **Placement:** uniform over stored bits: a block is chosen proportional to its bit count, then a tensor within it,
  then an element and bit. Default `fault_field="any"` (all 32 bits). Forcing `"exponent"` inflates severity; if used it
  must be disclosed (the summary records `fault_field`).
* **SBU/MBU mix:** 98.55 % single-bit, 1.45 % multi-bit clusters (max cluster size 4) per the Alsat-1 measurement.
* **Identical stream:** one seeded injector samples; every replica replays the same records. Differences between
  strategies are therefore *paired* per seed.
* **Persistence:** flips accumulate in memory until a repair overwrites them (SEUs in SRAM/DRAM persist).
  Transient *activation* faults are studied separately by `experiments/transient_faults.py`
  (measured: 0 % spurious restores, because T3 sees no stored-bit change).

## 4. Radiation environment

### 4.1 Upset rate calibration (measured)
Source: Y. Bentoutou, *Performance Comparison of Real Time EDAC Systems for Applications On-Board Small Satellites*,
IJECE 5(5), 2011 (full text verified). Alsat-1, 686 km SSO, commercial SRAM, Nov 2002 – Aug 2010:
265,649 upsets in 2,622 days → **4.04×10⁻⁷ upsets/bit/day**, 98.55 % single-bit.

The two configured rates (`background = 7.89e-8`, `SAA core = 5.94e-6` upsets/bit/day) are chosen so that the
orbit-averaged rate on the real Sentinel-2A track (≈ 79.9 SAA-core-equivalent min/day) reproduces 4.04×10⁻⁷.

### 4.2 Orbit and SAA (default `--orbit tle`)
1. SGP4 propagation of `orbits/sentinel2a.tle` (CelesTrak, NORAD 40697, ≈ 786–790 km SSO), sampled each minute.
2. The SAA is *defined physically* as low geomagnetic field: |B| from IGRF (`ppigrf`) at the satellite position.
   The upset rate ramps linearly from background at |B| ≥ 20,000 nT to the SAA-core rate at |B| ≤ 18,000 nT.
3. Upsets per step are Poisson with the rate integrated minute by minute.

Result: ≈ 80 SAA-core min/day and ≈ 82 % of upsets inside the SAA. `--orbit periodic` keeps the older synthetic
fixed passes (6 passes/day × 12 min).

### 4.3 Assumptions and gaps (disclose these)
* **ASSUMED** SAA-core/background ratio of 75.3× (carried over, not measured).
* **ASSUMED** transfer from Alsat-1 (686 km, 2002 SRAM) to Sentinel-2A (≈ 786 km, modern memory).
* **UNVERIFIED** Flying Laptop 73.41 % SAA upset fraction (paywalled; informational field only, not used for rates).
* **NOT DONE** AP8/AE9 trapped-flux model and device cross-sections; the IGRF |B| ramp is a proxy. A payload-specific
  rate needs SPENVIS/CREME96 plus device data.
* Solar-cycle variation (Alsat-1 yearly means rose from 41 to 119 upsets/day) is not modelled.

## 5. Detection method

| Tier | Signal | Calibration | Strength | Blind spot |
|---|---|---|---|---|
| T1 envelope | Conv/Linear output range, NaN/Inf | observed clean min/max on ≥ 100 chips, widened by 0.5×range | exponent-MSB explosions, NaN | faults that keep activations inside bounds |
| T2 BN drift | batch mean/std vs BN running stats | per-layer tolerance from clean batches; threshold 3.5 | mid-exponent / mantissa shifts | small weight errors that barely move statistics |
| T3 CRC-32 | stored parameter bytes | golden CRC at calibration | exact (any bit change) | only evaluated on demand (cost) |

### Filter-and-confirm
T1/T2 propose *candidate* blocks; **T3 confirms**. An exponent flip in one block corrupts activations in every
downstream layer, so many candidates appear; only the genuinely corrupted block(s) fail CRC, and cascade victims are
left alone. T3 also checks the block **upstream of the earliest candidate** because T1 hooks conv outputs and T2 hooks
BN inputs (a BN-parameter fault such as a sign flip on `layer3.0.bn2.running_var` → NaN first shows one block later).

### Known blind spot and the audit
A single sub-threshold fault can cost accuracy without any alarm (seed 1, step 49: a weight 0.106 → 1.7 cost
17 points). Hence **Aegis-v2 + CRC audit**: every `--audit-hours` (default 6) all blocks are CRC-scanned and any
mismatch restored. Report it **alongside** plain Aegis-v2.

## 6. Repair method

* Golden copy: zstd (level 3) over safetensors, one entry per block, in simulated rad-hard NVM. Lossless by default
  ⇒ restore is **bit-exact** and needs no BN recalibration.
* INT8 option (`--golden-int8`): 41.6 → 9.2 MB, −0.7 pts accuracy on 1,000 chips. Restored blocks are re-baselined
  individually in T3.
* **Never call `refresh_golden_checksums()` while any block may be corrupt**: it bakes the corruption into the T3
  reference and the fault becomes undetectable forever.

## 7. Energy and bandwidth accounting

| Item | Constant | Value | Status |
|---|---|---|---|
| Golden-store (SPI flash) read | `spi_flash_read_energy_pj_per_bit` | 15 pJ/bit | assumed |
| DRAM write on restore | `dram_write_energy_pj_per_bit` | 25 pJ/bit | assumed |
| DRAM read for T3 CRC scan | `dram_read_energy_pj_per_bit` | 20 pJ/bit | assumed (LPDDR4-class order of magnitude) |
| RF receive (uplink) | `radio_rx_energy_pj_per_bit` | 500 pJ/bit | assumed proxy |
| Inference | `inference_energy_mj_per_image` | 12.5 mJ | Jetson Xavier NX @ 15 W proxy |
| T1/T2 hook op | `monitor_energy_pj_per_op` | 6.9 pJ/op (= 12.5 mJ / 1.82 G MACs) | assumed (elementwise pass ≈ one MAC; likely pessimistic) |
| Uplink | `uplink_bandwidth_bps`, passes/day, pass length | 1 Mbps, 4 × 480 s ≈ 60 MB/pass | Chaohu-1 S-band |

T1/T2 hook compute is ~1.9 mJ per 16-image ResNet-18 batch, ≈ 138 mJ over 3 days, i.e. roughly ten full-reload
equivalents of memory energy, so it **matters** for the energy comparison. The report generator adds it analytically to
campaigns that predate in-simulation accounting.

Not charged: Ranger/BLINK per-inference compute (they move no bytes).

## 8. Metrics

* **Mean accuracy** – time-average over steps of top-1 accuracy on the labeled evaluation set (the primary metric).
* **Final accuracy** – accuracy at the last step (high variance; one bad fault can dominate).
* **MB/day** and **mJ/day** – bytes moved and energy per simulated day.
* **Paired Aegis-minus-baseline delta** – per-seed difference in mean accuracy, 95 % Student-t interval, wins count.
* **Scrub Pareto verdict** – is the strategy dominated by some scrubbing interval, and how much extra traffic would
  scrubbing need to match its accuracy (linear interpolation between swept intervals).
* **Statistics caveat** – intervals are over seeds (typically 3), so they are wide; do not over-interpret.

## 9. Validity invariants (each was a real bug)

1. **Never evaluate on `torch.randn`.** Random tensors are for unit-test plumbing only.
2. **T1 bounds = observed clean min/max + margin** over ≥ 100 calibration chips. A mean ± kσ envelope fires on every
   clean batch (heavy-tailed activations) and silently turns the detection gate into restore-everything.
3. **Calibration and evaluation chips must be disjoint** (`load_eurosat_splits`).
4. **Input size/normalisation/label order** must match the checkpoint (§2).
5. **Do not refresh golden checksums while corruption may exist** (§6).
6. **Alarm → block mapping must use `LayerRegistry.classify_key`** for parameter keys and module names.
7. **MobileNetV2 is trained only on chips outside the campaign splits**; `run_campaign.py` refuses other splits.
8. **Monitoring energy is charged on the monitored incoming batch only**, not on evaluation passes.
9. **Upstream block T3 check** (§5) – otherwise BN-parameter faults go unrepaired (accuracy stuck at 10 %).
10. **Disclose** `fault_field`, `golden_int8`, `recalibrate`, orbit model and "pre-fix" status with any number.

Results produced before 2026-10-21 (e.g. "98.9 % energy savings") violated invariants 1, 2 and 8 and are invalid.

## 10. Threats to validity

* Evaluation set is small (e.g. 15 chips/class = 150) → accuracy granularity ≈ 0.7 pt; few seeds.
* ResNet-18 training split unknown → clean accuracy may be optimistic (relative degradation unaffected).
* Radiation constants are transferred from a different mission/memory (§4.3).
* Faults are injected into parameters only; real devices also see control-logic and activation-buffer upsets.
* Energy figures are analytic proxies, not measured on flight hardware.
