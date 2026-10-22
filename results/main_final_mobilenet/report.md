# Campaign report: `main_final_mobilenet`

## Setup

- Architecture: `mobilenet_v2`
- Seeds: 0, 1, 2 (3); 3 simulated days each; 72 steps per seed
- Orbit: `tle`; SAA-core-equivalent minutes/day: 79.9
- Fault field: `any`; bit flips/day 29.1 (± 0.5); exponent-MSB flips/day 1.11 (± 0.96)
- Clean accuracy: 99.3%
- Golden store: lossy INT8 = False; BN recalibration = False

Intervals are 95% Student-t over 3 seeds.

## Accuracy, traffic and energy

| Strategy | Mean acc. (%) | Final acc. (%) | MB/day | Energy (mJ/day) |
|---|---|---|---|---|
| Aegis-v2 selective restore | 99.3 ± 0.0 | 99.3 ± 0.0 | 3.4 | 125.6 ± 2.5 |
| Aegis-v2 + CRC audit | 99.3 ± 0.0 | 99.3 ± 0.0 | 18.1 | 136.4 ± 2.5 |
| Ranger + Aegis-v2 | 99.3 ± 0.0 | 99.3 ± 0.0 | 1.7 | 124.7 ± 1.0 |
| Periodic scrubbing (12 h) | 87.2 ± 26.1 | 99.3 ± 0.0 | 16.9 | 5.7 ± 0.0 |
| Ground uplink reload | 93.3 ± 12.9 | 99.3 ± 0.0 | 3.5 | 141.5 ± 27.7 |
| Ranger clipping | 98.8 ± 1.4 | 96.7 ± 6.0 | 0.0 | not charged |
| BLINK re-centering | 79.5 ± 17.1 | 60.4 ± 44.9 | 0.0 | not charged |
| Unmitigated | 76.9 ± 51.7 | 39.8 ± 128.1 | 0.0 | not charged |
| Periodic scrubbing (1 h) | 99.3 ± 0.0 | n/a | 202.4 | 67.9 ± 0.0 |
| Periodic scrubbing (3 h) | 97.3 ± 4.7 | n/a | 67.5 | 22.6 ± 0.0 |
| Periodic scrubbing (6 h) | 96.4 ± 6.5 | n/a | 33.7 | 11.3 ± 0.0 |
| Periodic scrubbing (24 h) | 85.1 ± 31.8 | n/a | 8.4 | 2.8 ± 0.0 |

Ranger and BLINK move no bytes; their per-inference compute is not charged.

## Paired accuracy difference: Aegis-v2 minus baseline

Same fault stream per seed, so differences are paired.

| Baseline | Δ mean acc. (pts) | Seeds where Aegis-v2 higher |
|---|---|---|
| Aegis-v2 + CRC audit | +0.00 ± 0.00 | 0/3 |
| Ranger + Aegis-v2 | +0.00 ± 0.00 | 0/3 |
| Periodic scrubbing (12 h) | +12.12 ± 26.09 | 2/3 |
| Ground uplink reload | +5.99 ± 12.92 | 2/3 |
| Ranger clipping | +0.54 ± 1.37 | 2/3 |
| BLINK re-centering | +19.80 ± 17.14 | 3/3 |
| Unmitigated | +22.40 ± 51.71 | 2/3 |

## Bandwidth verdict

- **Aegis-v2 selective restore**: 99.3% at 3.4 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~202.4 MB/day to match it (59.80x the traffic).
- **Aegis-v2 + CRC audit**: 99.3% at 18.1 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~202.4 MB/day to match it (11.18x the traffic).
- **Ranger + Aegis-v2**: 99.3% at 1.7 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~202.4 MB/day to match it (118.74x the traffic).

Scrub-matching traffic is interpolated linearly between swept intervals.
