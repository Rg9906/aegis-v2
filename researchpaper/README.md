# Aegis-v2 research paper

IEEE two-column journal-format paper describing the whole project: problem, background, related work, fault model,
method, experimental setup, results (both final campaigns), lessons learned, threats to validity and future work.

| File | What it is |
|---|---|
| `aegis_v2_paper.pdf` | Compiled paper (11 pages) |
| `aegis_v2_paper.tex` | LaTeX source (`IEEEtran`, journal mode); Figs. 1 and 3 are TikZ inside the source |
| `references.bib` | Bibliography (47 entries) |
| `make_paper_figures.py` | Regenerates `figures/*.pdf` from `results/main_final*` and the frontend orbit export |
| `figures/` | Vector figures: accuracy over time, Pareto, costs, fault timeline, orbit/SAA map |

## Build

Figures (needs the final campaigns in `results/` and `frontend/public/data/main_final.json`):

```bash
python researchpaper/make_paper_figures.py
```

PDF (TeX Live / TinyTeX / MiKTeX with `IEEEtran`, `tikz`, `booktabs`, `algorithms`):

```bash
cd researchpaper
pdflatex aegis_v2_paper && bibtex aegis_v2_paper && pdflatex aegis_v2_paper && pdflatex aegis_v2_paper
```

Overleaf: upload the `.tex`, `.bib` and the `figures/` folder; the default compiler works.

## Before submitting

* **Affiliation.** The author footnote has a `[Department, Institution]` placeholder.
* **Verify recent references.** Entries marked `% VERIFY` in `references.bib` (BLINK, CoG, SHNN, SENTRY, ZEDAC,
  Safe-NEureka, the ECC-for-DNN preprint, WeightSentry, "AI Infrastructure in Space") are 2026 preprints or proceedings
  taken from the project's prior-art map. Their titles and identifiers are as listed there; authors were not verified
  and are omitted rather than guessed. RedNet (arXiv 2407.11853) also has no author list for the same reason.
* **Numbers.** Every number comes from `results/main_final/report.md`, `results/main_final_mobilenet/report.md`, the
  seed JSON files or `results/transient_faults.json`. If you rerun the campaigns, regenerate the figures and update the
  tables in Section VII.
* **Claims to keep honest.** The advantage is bandwidth and availability, not energy (Section VII-D). Three seeds and
  150 evaluation chips give wide intervals; the MobileNetV2 paired differences are mostly not significant.
