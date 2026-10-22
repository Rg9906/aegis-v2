# User Guide

Practical instructions for installing, running and reading everything in this repository. Commands assume the
repository root as the working directory.

## 1. Requirements

* Python ≥ 3.10 (the tested interpreter is **Python 3.12**). Dependencies: `torch`, `torchvision`, `safetensors`,
  `zstandard`, `numpy`, `scipy`, `huggingface_hub`, `streamlit`, `matplotlib`, `sgp4`, `ppigrf` (+ `pytest`).
  The Streamlit dashboard additionally imports `altair` and `pandas`, which `streamlit` pulls in transitively.
* Node.js + npm for the optional React frontend.
* A GPU is optional: CPU campaigns take ~35 s per simulation step with 9 replicas; GPU runs use the Colab notebook.

### Windows machine notes (author's setup)
* Use `C:\Users\RG Saran Vishakan\AppData\Local\Microsoft\WindowsApps\python3.12.exe`; the default `python` on `PATH`
  is an unrelated venv without torch.
* Python `requests` fails SSL to huggingface.co here; `curl` works.
* Set `HF_HUB_OFFLINE=1` so the cached `cm93/resnet18-eurosat` weights load with `local_files_only`.

## 2. Install

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```

## 3. Data

EuroSAT RGB must be at `data/eurosat/2750/<ClassName>/*.jpg` (the folder is gitignored).

```bash
curl -L -o data/eurosat/EuroSAT.zip https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip
# extract so that data/eurosat/2750/AnnualCrop/..., Forest/..., etc. exist
```

The real-data test skips automatically if `data/` is missing.

## 4. Commands

| Task | Command |
|---|---|
| Tests (~1 min) | `python -m pytest -q` |
| Demo | `python demo.py` |
| Campaign | `python experiments/run_campaign.py [flags]` |
| Figures | `python experiments/make_figures.py --name <campaign>` |
| Report | `python experiments/write_report.py --name <campaign>` |
| Train MobileNetV2 | `python experiments/train_mobilenet.py --epochs 3 --device cuda` |
| Transient faults | `python experiments/transient_faults.py` |
| Streamlit dashboard | `streamlit run dashboard/app.py` |
| Frontend data export | `python experiments/export_frontend_data.py --name <campaign>` |
| Frontend dev / build | `cd frontend && npm install && npm run dev` / `npm run build` |

### 4.1 `run_campaign.py` flags

| Flag | Default | Meaning |
|---|---|---|
| `--days` | 3.0 | Simulated mission length |
| `--step-minutes` | 60 | Simulation step (one monitored batch + one evaluation per step) |
| `--seeds` | `0` | One or more seeds; each is an independent fault stream (`seed_<n>.json`) |
| `--eval-per-class` | 20 | Labeled evaluation chips per class |
| `--calib-per-class` | 10 | Unlabeled calibration chips per class (must give ≥ 100 chips for T1 bounds) |
| `--fault-field` | `any` | `any\|exponent\|mantissa\|sign`; non-default values must be disclosed |
| `--scrub-interval-hours` | 12 | Interval of the headline periodic-scrubbing baseline |
| `--scrub-sweep-hours` | none | Extra scrub replicas (e.g. `1 3 6 24`) for the accuracy-vs-bandwidth Pareto |
| `--orbit` | `tle` | `tle` (SGP4 + IGRF SAA) or `periodic` (synthetic passes) |
| `--tle` | `orbits/sentinel2a.tle` | Alternative TLE file |
| `--golden-int8` | off | INT8 golden store (lossy, 4× smaller) |
| `--recalibrate` | off | Enable post-restore BN recalibration (charged inference energy) |
| `--audit-hours` | 6 | Period of the full CRC audit (`aegis_audit`); `0` disables the periodic scan |
| `--arch` | `resnet18` | `resnet18` or `mobilenet_v2` |
| `--checkpoint` | per arch | Weights file |
| `--device` | `auto` | `auto\|cpu\|cuda` |
| `--name` | `main` | Output folder name under `--out-dir` |
| `--out-dir` | `results` | Output root |

Reference run (≈ 35 s/step on CPU):

```bash
python experiments/run_campaign.py --days 3 --seeds 0 1 2 --eval-per-class 15 \
    --scrub-sweep-hours 1 3 6 24 --name main
```

Outputs: `results/<name>/seed_<n>.json` (summary + per-step telemetry) and `results/<name>/campaign.json`.

### 4.2 `train_mobilenet.py` flags
`--epochs 3`, `--batch-size 64`, `--lr 1e-3`, `--exclude-splits` (the `(calib,eval)` per-class configs to hold out;
the campaign refuses splits outside the held-out set), `--no-imagenet` (random init, smoke tests only),
`--max-train` (cap, smoke tests only), `--device`, `--data-dir`, `--out` (default checkpoint under gitignored
`checkpoints/`). It writes a sidecar listing the held-out chips.

Then: `python experiments/run_campaign.py ... --arch mobilenet_v2 --name <campaign>`.

### 4.3 Other scripts
* `transient_faults.py` – `--trials 60 --bit 30 --batch 16 --seed 0 --out results/transient_faults.json`. Injects
  transient activation faults and counts spurious restores (committed result: 0 %).
* `make_figures.py` / `write_report.py` / `export_frontend_data.py` – `--name`, `--results-dir` (and `--out-dir` for export).

## 5. Typical workflows

**A. Reproduce the main results (CPU).** Install → data → `pytest` → run the reference campaign → figures → report.

**B. Colab / GPU.** Open `notebooks/aegis_colab.ipynb` and choose a **T4 GPU** runtime. Cells: set `GITHUB_REPO`
(`Rg9906/aegis-v2`), `CAMPAIGN` flags and `NAME` (currently `main_final`); clone (private repo: token in Colab secret
`GH_TOKEN`); install `sgp4 ppigrf zstandard safetensors huggingface_hub`; download EuroSAT; run
`tests/test_correctness_fixes.py` and `tests/test_orbit_model.py`; run the campaign with `--device cuda`; figures; report;
download `results_<NAME>.zip` and unzip into local `results/`. An optional second section fine-tunes MobileNetV2
(about 10 min on a T4), runs `<NAME>_mobilenet`, and bundles `checkpoints/` into the results zip.

**C. Second architecture.** `train_mobilenet.py` (GPU) → `run_campaign.py --arch mobilenet_v2`.

**D. Presentation.** `export_frontend_data.py --name <campaign>` → `cd frontend && npm install && npm run dev`
(http://localhost:5173) or `npm run build` for a static site in `frontend/dist/`.

**E. Interactive exploration.** `streamlit run dashboard/app.py`.

## 6. Using the library directly

```python
from aegis.data.eurosat import load_eurosat_splits
from aegis.models.factory import load_backbone
from aegis.simulation.mission_sim import MissionSimulationHarness

splits = load_eurosat_splits()                      # disjoint calibration / evaluation tensors
model  = load_backbone("resnet18")
sim = MissionSimulationHarness(
    model, splits.eval_images, splits.eval_labels,
    calib_images=splits.calib_images, random_seed=0,
    orbit="tle", scrub_sweep_hours=(1, 6),
)
telemetry = sim.run_simulation(total_days=1.0, progress=True)
print(sim.get_campaign_summary()["mean_accuracy"])
```

`load_eurosat_splits` defaults to 4 calibration chips/class; pass `calib_per_class=10` (the `run_campaign.py`
default) so T1 bounds are built from ≥ 100 clean chips. Evaluation chips default to 20/class.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: torch` | Wrong interpreter; use the Python 3.12 listed above |
| SSL error contacting huggingface.co | Use `curl` for downloads; set `HF_HUB_OFFLINE=1` |
| ≈ 18 % or chance accuracy | Wrong input size (must be 224 px) or alphabetical labels |
| Aegis restores on every step | T1 envelope miscalibrated (needs ≥ 100 clean chips, min/max bounds) |
| Accuracy stuck at 10 % after BN fault | Campaign predates the upstream-block T3 check ("pre-fix") |
| Campaign refuses a MobileNetV2 split | Split not inside the checkpoint's held-out sidecar; retrain with that split excluded |
| Frontend export asserts on flip counts | Fault-stream replay disagrees with recorded telemetry; re-run campaign with the same code version |
| Dashboard shows fewer strategies | Older campaign JSON lacks newer strategies (handled by `strategies_present`) |
