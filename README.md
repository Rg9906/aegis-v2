# Project Aegis-v2

**Detection-gated, layer-selective restore for a radiation-corrupted onboard Earth-observation classifier.**

Aegis-v2 is a research prototype that asks one question: *when cosmic-ray / South Atlantic Anomaly (SAA) bit flips
corrupt the weights of a neural network running on a LEO satellite, how can we repair it with the least
downlink bandwidth, memory traffic and energy, without labels and without waiting for a ground contact?*

The testbed is a ResNet-18 trained on EuroSAT (Sentinel-2 RGB land-use chips), flown through a simulated
multi-day orbit of the real Sentinel-2A ground track. Eight mitigation strategies see the **identical, seeded
fault stream** and are compared on accuracy, bytes moved and energy.

> **Status: research prototype** (phases 1-4 done; final campaigns run 2026-10-22). Results dated before 2026-10-21 were produced with random-noise inputs and an
> always-on detection gate and are **invalid**; do not cite them. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
> for the validity rules and every modelling assumption.

---

## The idea in 60 seconds

```
 incoming (unlabeled) image batch
            │
            ▼
   ┌───────────────────┐   cheap, every batch, label-free
   │ T1 activation     │── conv outputs outside clean [min,max]+margin, NaN/Inf
   │ T2 BatchNorm drift│── batch stats vs. BN running stats (z-score style)
   └─────────┬─────────┘
             │ candidate blocks (+ the block just upstream)
             ▼
   ┌───────────────────┐   deterministic confirmation
   │ T3 CRC-32 of block│── only blocks whose stored bits really changed
   └─────────┬─────────┘
             │ confirmed-corrupt blocks only
             ▼
   ┌───────────────────┐   rad-hard companion memory
   │ Golden store      │── zstd + safetensors, one entry per block
   │ selective restore │── bit-exact hot-patch of just those blocks
   └───────────────────┘
```

* **Detect behaviourally** (T1/T2) so no labelled data and no ground contact is needed.
* **Confirm deterministically** (T3) so a noisy detector never triggers a needless restore.
* **Restore only the corrupt blocks** from a compressed, lossless golden copy.
* **Audit periodically** (optional "Aegis-v2 + CRC audit") to catch faults too small for T1/T2 to see.

## Strategies compared

| Key (summary JSON) | Label | What it does |
|---|---|---|
| `unmitigated` | Unmitigated | Takes the faults, does nothing. |
| `ranger_clipping` | Ranger clipping | Clamps activations to calibrated bounds (Chen et al., DSN 2021). No repair. |
| `blink_recentering` | BLINK re-centering | Re-centres BN activations in-situ. No repair. |
| `periodic_scrubbing` | Periodic scrubbing (default 12 h) | Blind full-model reload on a timer. Optional sweep (1/3/6/24 h) for the Pareto curve. |
| `aegis_v2` | **Aegis-v2 selective restore** | T1/T2 → T3 → restore only confirmed blocks. |
| `ranger_aegis` | Ranger + Aegis-v2 | Aegis-v2 plus Ranger clamping (monitor hooks see raw activations). |
| `ground_uplink` | Ground uplink reload | Same detection, but repairs only during ground-contact windows over a ~1 Mbps link. |
| `aegis_audit` | Aegis-v2 + CRC audit | Aegis-v2 plus a full CRC-32 scan of all blocks every `--audit-hours` (default 6). |

## Quick start

All commands run from the repository root, inside the project venv (`.venv`, Python 3.12). On the
author's machine the default `python` on `PATH` is an unrelated venv without torch, so activate `.venv` first.

```bash
# 1. Create the venv and install dependencies
python3.12 -m venv .venv
.venv\Scripts\Activate.ps1                 # PowerShell; bash: source .venv/Scripts/activate
pip install -r requirements.txt            # or: pip install -e ".[dev]"

# 2. Get the dataset (gitignored). curl works; python `requests` fails SSL to HF on this machine.
mkdir -p data/eurosat
curl -L -o data/eurosat/EuroSAT.zip https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip
#    then extract so that images are under data/eurosat/2750/<ClassName>/*.jpg

# 3. Verify the install (~1 min)
python -m pytest -q

# 4. Run the quick demo
HF_HUB_OFFLINE=1 python demo.py

# 5. Run a campaign, then build figures + report
HF_HUB_OFFLINE=1 python experiments/run_campaign.py --days 3 --seeds 0 1 2 --eval-per-class 15 \
    --scrub-sweep-hours 1 3 6 24 --name main
python experiments/make_figures.py --name main
python experiments/write_report.py --name main

# 6. Presentation: build the UI once, then one command runs the live console + briefing
cd frontend && npm install && npm run build && cd ..
python -m aegis.console --open          # http://127.0.0.1:8765/#/console
```

The live console drives the real model (strike bits, watch detection and repair, fly live missions, classify
chips); see [docs/FRONTEND.md](docs/FRONTEND.md). Keep the laptop plugged in and ~1.5 GB of RAM free.

The pretrained weights (`cm93/resnet18-eurosat`) are expected in the Hugging Face cache and load with
`local_files_only`; set `HF_HUB_OFFLINE=1` for scripts.

## Headline results (final campaigns, 2026-10-22)

Colab T4, 3 seeds x 3 simulated days, real Sentinel-2A orbit, 150 evaluation chips, audit strategy included.
Intervals are 95 % Student-t over 3 seeds, so they are wide.

| Campaign | Strategy | Mean accuracy | MB/day | mJ/day |
|---|---|---|---|---|
| `main_final` (ResNet-18, clean 89.3 %) | **Aegis-v2 selective restore** | **87.4 ± 8.3 %** | 188.1 | 144.1 |
| | Aegis-v2 + CRC audit | 88.9 ± 1.7 % | 279.6 | 202.3 |
| | Periodic scrubbing 12 h | 37.0 ± 27.0 % | 83.2 | 27.9 |
| | Ground uplink reload | 66.2 ± 10.2 % | 121.5 | 634.5 |
| | Unmitigated | 14.0 ± 9.8 % | 0 | n/a |
| `main_final_mobilenet` (MobileNetV2, clean 99.3 %) | **Aegis-v2 selective restore** | **99.3 ± 0.0 %** | 3.4 | 125.6 |
| | Periodic scrubbing 12 h | 87.2 ± 26.1 % | 16.9 | 5.7 |
| | Unmitigated | 76.9 ± 51.7 % | 0 | n/a |

**The claim is bandwidth, not energy.** Matching Aegis-v2's accuracy by scrubbing alone needs about 889 MB/day on
ResNet-18 (4.7x) and about 202 MB/day on MobileNetV2 (about 60x). Aegis-v2 does **not** save energy: T1/T2 monitor hook
compute puts it at about 125-144 mJ/day versus about 6-28 mJ/day for 12 h scrubbing. MobileNetV2 seed 2 had no damaging
fault, and 3 seeds / 150 chips give wide intervals. `results/main_gpu` is an older pre-fix campaign: do not cite it.
Details: [docs/RESULTS.md](docs/RESULTS.md).

## Documentation map

| Document | Read it for |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Package layout, data flow, every module and class, the per-step simulation protocol |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Fault model, orbit/radiation model, detectors, energy accounting, assumptions, validity invariants |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Setup, every CLI flag, workflows (campaign → figures → report → frontend), Colab |
| [docs/RESULTS.md](docs/RESULTS.md) | Result artefacts, JSON schemas, how to read the report, current numbers and caveats |
| [docs/TESTING.md](docs/TESTING.md) | Test suite map and what each regression test protects |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Extending Aegis (new baseline, architecture, monitor tier), pitfalls, known limitations, roadmap |
| [docs/FRONTEND.md](docs/FRONTEND.md) | The React mission dossier: data export, structure, build and deploy |
| [docs/STATUS.md](docs/STATUS.md) | Dated project history: what was built, fixed and invalidated |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Radiation, ML and project terminology |
| [CLAUDE.md](CLAUDE.md) | Short operating notes and invariants for AI-assisted work |
| [AEGIS_RESEARCH_STUDY_AND_PROTOTYPE_BLUEPRINT.md](AEGIS_RESEARCH_STUDY_AND_PROTOTYPE_BLUEPRINT.md) | Original design rationale |
| `Project Aegis_ Prior-Art Map ... .md` | Prior-art map and gap analysis |

## Repository layout

```
aegis/            core library (fault engine, monitors, repair, baselines, simulation, data, models)
experiments/      CLI entry points: campaign, figures, report, frontend export, MobileNetV2 training, transient faults
aegis/console/    Live console engine: FastAPI + WebSocket over the real pipeline (`python -m aegis.console`)
frontend/         React + Vite app: Briefing (recorded campaigns) + Live console (real engine)
notebooks/        Colab notebook for GPU campaigns
orbits/           sentinel2a.tle (CelesTrak) used by the SGP4 orbit model
results/          campaign outputs (seed_*.json, campaign.json, report.md, figures/)
tests/            pytest suite
demo.py           single-run end-to-end demonstration
```

## Honest limitations

* Upset rates are calibrated to the *measured* Alsat-1 mean (4.04e-7 upsets/bit/day), but the SAA-core/background
  ratio (75.3×) and the transfer from a 2002 SRAM at 686 km to Sentinel-2A's memory at 786 km are **assumed**.
* No AP8/AE9 trapped-flux model or device cross-section; the IGRF |B| ramp is a proxy.
* The ResNet-18 checkpoint's training split is unknown, so clean accuracy may be optimistic (relative degradation
  is unaffected). The MobileNetV2 path has a verified held-out split.
* Aegis-v2 is not an energy saver (monitor compute dominates); only the bandwidth claim is supported.
* Ranger / BLINK per-inference compute is not charged to their energy.
* Behavioural detection cannot see sub-threshold faults; that is why the audit strategy exists.

## Licence and citation

No licence file is present in the repository; add one before redistributing. `pyproject.toml` lists the authors as
"Project Aegis Research Team".
