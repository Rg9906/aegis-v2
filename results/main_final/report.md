# Campaign report: `main_final`

## Setup

- Architecture: `resnet18`
- Seeds: 0, 1, 2 (3); 3 simulated days each; 72 steps per seed
- Orbit: `tle`; SAA-core-equivalent minutes/day: 79.9
- Fault field: `any`; bit flips/day 147.1 (± 16.6); exponent-MSB flips/day 4.89 (± 2.08)
- Clean accuracy: 89.3%
- Golden store: lossy INT8 = False; BN recalibration = False

Intervals are 95% Student-t over 3 seeds.

## Accuracy, traffic and energy

| Strategy | Mean acc. (%) | Final acc. (%) | MB/day | Energy (mJ/day) |
|---|---|---|---|---|
| Aegis-v2 selective restore | 87.4 ± 8.3 | 83.6 ± 24.9 | 188.1 | 144.1 ± 35.8 |
| Aegis-v2 + CRC audit | 88.9 ± 1.7 | 89.3 ± 0.0 | 279.6 | 202.3 ± 20.3 |
| Ranger + Aegis-v2 | 87.4 ± 8.3 | 83.6 ± 24.9 | 143.0 | 120.5 ± 35.6 |
| Periodic scrubbing (12 h) | 37.0 ± 27.0 | 89.3 ± 0.0 | 83.2 | 27.9 ± 0.0 |
| Ground uplink reload | 66.2 ± 10.2 | 52.4 ± 91.4 | 121.5 | 634.5 ± 167.3 |
| Ranger clipping | 38.5 ± 56.4 | 27.1 ± 43.1 | 0.0 | not charged |
| BLINK re-centering | 34.0 ± 42.0 | 13.6 ± 16.8 | 0.0 | not charged |
| Unmitigated | 14.0 ± 9.8 | 10.0 ± 0.0 | 0.0 | not charged |
| Periodic scrubbing (1 h) | 89.3 ± 0.0 | n/a | 998.0 | 334.6 ± 0.0 |
| Periodic scrubbing (3 h) | 77.6 ± 8.2 | n/a | 332.7 | 111.5 ± 0.0 |
| Periodic scrubbing (6 h) | 66.4 ± 23.1 | n/a | 166.3 | 55.8 ± 0.0 |
| Periodic scrubbing (24 h) | 22.0 ± 13.1 | n/a | 41.6 | 13.9 ± 0.0 |

Ranger and BLINK move no bytes; their per-inference compute is not charged.

## Paired accuracy difference: Aegis-v2 minus baseline

Same fault stream per seed, so differences are paired.

| Baseline | Δ mean acc. (pts) | Seeds where Aegis-v2 higher |
|---|---|---|
| Aegis-v2 + CRC audit | -1.53 ± 6.57 | 0/3 |
| Ranger + Aegis-v2 | +0.00 ± 0.00 | 0/3 |
| Periodic scrubbing (12 h) | +50.45 ± 23.56 | 3/3 |
| Ground uplink reload | +21.17 ± 5.31 | 3/3 |
| Ranger clipping | +48.91 ± 48.22 | 3/3 |
| BLINK re-centering | +53.39 ± 36.36 | 3/3 |
| Unmitigated | +73.41 ± 4.76 | 3/3 |

## Bandwidth verdict

- **Aegis-v2 selective restore**: 87.4% at 188.1 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~889.1 MB/day to match it (4.73x the traffic).
- **Aegis-v2 + CRC audit**: 88.9% at 279.6 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~975.4 MB/day to match it (3.49x the traffic).
- **Ranger + Aegis-v2**: 87.4% at 143.0 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~889.1 MB/day to match it (6.22x the traffic).

Scrub-matching traffic is interpolated linearly between swept intervals.
