# Glossary

## Radiation and orbit
| Term | Meaning |
|---|---|
| **SEU** | Single-event upset: a particle strike flips a stored bit (no permanent damage). The model's weights sit in memory, so SEUs corrupt them until repaired. |
| **SBU / MBU** | Single-bit / multi-bit upset. Alsat-1 measured 98.55 % SBU; MBU clusters here have up to 4 bits. |
| **SAA** | South Atlantic Anomaly: region where the geomagnetic field is weak and trapped protons reach low altitude, so upset rates spike. Defined here as low |B| from IGRF. |
| **LEO / SSO** | Low Earth orbit / sun-synchronous orbit (Sentinel-2A: about 786 km). |
| **TLE / SGP4** | Two-line element set (orbit description) and the propagator that turns it into positions over time. |
| **IGRF, |B|** | International Geomagnetic Reference Field; |B| is total field strength in nT. Rate ramps from background at 20,000 nT to SAA-core at 18,000 nT. |
| **AP8/AE9** | Standard trapped proton/electron flux models. Not implemented (future work). |
| **Cross-section** | Device sensitivity to particle strikes; not modelled per device. |
| **Upsets/bit/day** | Rate unit. Calibrated mean: 4.04e-7 (Alsat-1). |

## Floating point
| Term | Meaning |
|---|---|
| **IEEE-754 float32** | 1 sign bit, 8 exponent bits (bits 30-23), 23 mantissa bits. |
| **Exponent MSB (bit 30)** | Flipping it can multiply a value by about 2^128, causing activation explosions or NaN/Inf. Most damaging. |
| **Sub-threshold fault** | A flip that changes accuracy but stays inside T1/T2 limits, so no behavioural alarm fires. |

## Model and data
| Term | Meaning |
|---|---|
| **EuroSAT** | 27,000 labelled 64x64 Sentinel-2 RGB chips, 10 land-use classes. |
| **Block** | Unit of detection/restore/energy accounting: `stem`, `layerX_blockY`, `classifier` (ResNet); `features_blockN` (MobileNetV2). |
| **Calibration chips** | Unlabeled clean chips used to set T1/T2 thresholds (>= 100, disjoint from evaluation). |
| **Evaluation chips** | Labeled chips used only to score accuracy. |
| **Held-out split** | Chips excluded from MobileNetV2 training so evaluation is honest. |
| **BN** | Batch Normalization; running mean/var are stored buffers that faults can corrupt. |

## Aegis components
| Term | Meaning |
|---|---|
| **T1** | Activation envelope monitor (range + NaN/Inf check on conv outputs). |
| **T2** | BatchNorm drift monitor (batch stats vs running stats). |
| **T3** | CRC-32 checksum monitor over block bytes; deterministic confirmation. |
| **Filter-and-confirm** | T1/T2 propose candidate blocks; T3 confirms which are truly corrupt. |
| **Primary suspect** | Most upstream confirmed block in forward order. |
| **Golden store** | Compressed (zstd + safetensors) clean copy of every block in simulated rad-hard memory. |
| **Selective restore** | Overwriting only confirmed-corrupt blocks from the golden store (bit-exact when lossless). |
| **CRC audit** | Periodic full T3 scan of all blocks to catch silent faults ("Aegis-v2 + CRC audit"). |
| **Recalibration** | Optional BN statistic refresh after a lossy restore; off by default. |
| **Pre-fix** | Campaign predating the upstream-localisation fix and CRC audit; not citable. |

## Baselines
| Term | Meaning |
|---|---|
| **Ranger** | Clamp activations to calibrated bounds (DSN 2021); masks, never repairs. |
| **BLINK** | In-situ BN re-centering; masks, never repairs. |
| **Periodic scrubbing** | Blind full-model reload every N hours. |
| **Ground uplink** | Same detection, but replacement weights arrive only in ground contacts (about 1 Mbps, 4 x 8 min/day). |

## Metrics and statistics
| Term | Meaning |
|---|---|
| **Mean accuracy** | Time-average of per-step accuracy; primary metric. |
| **MB/day, mJ/day** | Repair traffic and energy per simulated day. |
| **Paired delta** | Per-seed (Aegis minus baseline) accuracy difference, valid because all strategies share a fault stream. |
| **95 % Student-t interval** | Confidence interval over seeds; wide at n = 3. |
| **Pareto / dominated** | A strategy is dominated if another gets higher accuracy at equal or lower traffic. |
| **Bandwidth verdict** | Traffic scrubbing needs to match a strategy's accuracy (linear interpolation across the sweep). |
