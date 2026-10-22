# Aegis-v2 research paper

IEEE two-column journal-format paper describing the whole project: problem, background, related work, fault model and
simulator, method, experimental setup, results on both final campaigns, an exact ablation, a single-fault detection
study, class-confusion analysis, energy accounting, lessons learned, threats to validity and future work.

| File | What it is |
|---|---|
| `aegis_v2_paper.pdf` | Compiled paper |
| `aegis_v2_paper.tex` | LaTeX source (`IEEEtran`, journal mode); the concept, IEEE-754, simulator and architecture diagrams are TikZ inside the source |
| `references.bib` | Bibliography; recent preprints checked against their arXiv/proceedings records (Oct. 2026) |
| `make_paper_figures.py` | Main figures: accuracy over time, Pareto, costs, fault timeline, orbit/SAA map |
| `make_paper_tables.py` | Every data table (`tables/*.tex`) plus the secondary figures: per-seed, rate profile, traffic over time, energy breakdown, ablation, detection study, class confusion |
| `figures/`, `tables/` | Generated outputs; do not edit by hand |

## Build

Data (from the repository root; the detection study is CPU-heavy, roughly 30 s per trial on a laptop CPU):

```bash
python experiments/detection_study.py --arch mobilenet_v2 --trials-per-field 30
python experiments/detection_study.py --arch resnet18 --trials-per-field 30
python researchpaper/make_paper_figures.py
python researchpaper/make_paper_tables.py
```

PDF (TeX Live / TinyTeX / MiKTeX with `IEEEtran`, `tikz`, `booktabs`, `algorithms`):

```bash
cd researchpaper
pdflatex aegis_v2_paper && bibtex aegis_v2_paper && pdflatex aegis_v2_paper && pdflatex aegis_v2_paper
```

Overleaf: upload the `.tex`, `.bib`, `figures/` and `tables/`; the default compiler works.

## Sources of every number

* `results/main_final/`, `results/main_final_mobilenet/`: the final campaigns (Tables X-XV, XVIII; Figs. 7-14).
* `results/main_gpu/`: the pre-fix campaign on identical fault streams, used only for the upstream-rule ablation.
* `results/detection_study_*.json`: single-fault detection study and class confusion.
* `results/transient_faults.json`: transient activation faults.
* `frontend/public/data/main_final*.json`: per-step fault records and the orbit/field export.

## Before submitting

* **Affiliation.** The author footnote has a `[Department, Institution]` placeholder.
* **References.** Two items from the prior-art map (SHNN in Scientific Reports, WeightSentry) and a U.S. patent whose
  record did not match its claimed content were dropped because their records could not be verified.
* **Claims to keep honest.** The advantage is bandwidth and availability, not energy. Three seeds and 150 evaluation
  chips give wide intervals; the MobileNetV2 paired differences are mostly not significant.
