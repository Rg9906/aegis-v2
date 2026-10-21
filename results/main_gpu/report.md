# Campaign report: `main_gpu`

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
| Aegis-v2 selective restore | 87.0 ± 9.9 | 83.6 ± 24.9 | 160.4 | 128.4 ± 29.6 |
| Ranger + Aegis-v2 | 74.8 ± 62.6 | 83.6 ± 24.9 | 175.4 | 146.4 ± 214.5 |
| Periodic scrubbing (12 h) | 37.0 ± 27.0 | 89.3 ± 0.0 | 83.2 | 27.9 ± 0.0 |
| Ground uplink reload | 66.2 ± 10.2 | 52.4 ± 91.4 | 109.4 | 573.0 ± 206.3 |
| Ranger clipping | 38.5 ± 56.4 | 27.1 ± 43.1 | 0.0 | not charged |
| BLINK re-centering | 34.0 ± 42.0 | 13.6 ± 16.8 | 0.0 | not charged |
| Unmitigated | 14.0 ± 9.8 | 10.0 ± 0.0 | 0.0 | not charged |
| Periodic scrubbing (1 h) | 89.3 ± 0.0 | n/a | 998.0 | 334.6 ± 0.0 |
| Periodic scrubbing (3 h) | 77.6 ± 8.2 | n/a | 332.7 | 111.5 ± 0.0 |
| Periodic scrubbing (6 h) | 66.4 ± 23.1 | n/a | 166.3 | 55.8 ± 0.0 |
| Periodic scrubbing (24 h) | 22.0 ± 13.1 | n/a | 41.6 | 13.9 ± 0.0 |

Ranger and BLINK move no bytes; their per-inference compute is not charged.
Monitored strategies include T1/T2 hook compute added analytically (campaign predates its in-simulation accounting).

## Paired accuracy difference: Aegis-v2 minus baseline

Same fault stream per seed, so differences are paired.

| Baseline | Δ mean acc. (pts) | Seeds where Aegis-v2 higher |
|---|---|---|
| Ranger + Aegis-v2 | +12.25 ± 52.70 | 1/3 |
| Periodic scrubbing (12 h) | +50.09 ± 23.19 | 3/3 |
| Ground uplink reload | +20.81 ± 5.20 | 3/3 |
| Ranger clipping | +48.54 ± 46.68 | 3/3 |
| BLINK re-centering | +53.02 ± 35.41 | 3/3 |
| Unmitigated | +73.04 ± 4.91 | 3/3 |

## Bandwidth verdict

- **Aegis-v2 selective restore**: 87.0% at 160.4 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~868.4 MB/day to match it (5.42x the traffic).
- **Ranger + Aegis-v2**: 74.8% at 175.4 MB/day.
  Not dominated by any scrubbing interval.
  Scrubbing needs ~291.3 MB/day to match it (1.66x the traffic).

Scrub-matching traffic is interpolated linearly between swept intervals.
