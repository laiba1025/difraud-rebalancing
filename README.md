# Rebalancing and Cross-Dataset Generalization: DiFrauD and UNIDECOR

This project tests four recent rebalancing techniques with three classifiers on the DiFrauD dataset (per domain and pooled), and checks how well the detectors transfer between DiFrauD and UNIDECOR in both directions, using frozen and retrained models.

## Setup

Python 3.11.9. Run from the project root (Windows paths shown; on Linux/macOS use `.venv/bin/python`).

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

**DiFrauD** is downloaded automatically from HuggingFace (`redasers/difraud`, commit `f8fa04a`) on first run.

**UNIDECOR** must be built with its authors' scripts:

```
git clone -b unidecor_v1.0 https://bitbucket.org/aswathyve/unifieddeceptioncorpus.git unidecor_repo
cd unidecor_repo
../.venv/Scripts/python download.py --yes
PYTHONUTF8=1 ../.venv/Scripts/python get_consolidated_data.py
cd ..
```

If `download.py` fails for CROSS_CULT_DE, OPEN_DOMAIN or TRIAL_DE (SSL error), download them manually from https://web.eecs.umich.edu/~mihalcea/downloads.html into `unidecor_repo/datasets/downloaded/`.

## How to run

Run from `src/`, in this order:

| # | Script | Output |
|---|---|---|
| 1 | `inspect_data.py` | `results/data_stats/` |
| 2 | `check_cross_duplicates.py` | `results/data_stats/` |
| 3 | `run_in_domain.py --techniques baseline` | `results/in_domain/` |
| 4 | `check_rebalancing.py` | `results/rebalancing_checks/` |
| 5 | `run_in_domain.py --techniques simplicial mgs meb spe --reuse-features --out results/in_domain_rebalancing --baseline results/in_domain/summary.csv` | `results/in_domain_rebalancing/` |
| 6 | `run_cross_dataset.py` | `results/cross_dataset/` |
| 7 | `make_summary_tables.py` | `results/summary_tables/` |

Example: `PYTHONUTF8=1 ../.venv/Scripts/python inspect_data.py`

Step 6 reuses the models saved in steps 3 and 5.

## Code

| File | Purpose |
|---|---|
| `data_loading.py` | Loads both datasets, maps labels (deceptive = 1), creates UNIDECOR splits |
| `features.py` | TF-IDF (1–2-grams) + SVD to 300 dimensions |
| `classifiers.py` | Logistic regression, linear SVM, XGBoost |
| `rebalancing.py` | The four rebalancing techniques |
| `metrics.py` | Macro-F1, deceptive precision/recall/F1, AUC-ROC |
| `run_in_domain.py` | In-domain experiments |
| `run_cross_dataset.py` | Cross-dataset experiments |
| `make_summary_tables.py` | Final tables |

## Setup details

- **Data:** 7 DiFrauD domains (95,854 texts) with official splits; 11 of 14 UNIDECOR components (41,968 English texts) with stratified 80/10/10 splits (Diplomacy keeps its official split).
- **Features:** fitted on training data only.
- **Classifiers:** default settings, threshold 0.5.
- **Seeds:** 42, 43, 44.
- **Positive class:** deceptive.

## Rebalancing techniques

| Technique | Source | Implementation |
|---|---|---|
| Simplicial SMOTE | Kachan et al., KDD 2025 | Own implementation (`k=10`, `p=3`) |
| MGS | Sakho et al., AISTATS 2026 | Based on the authors' `mgs-grf` package (defaults) |
| MEB-SMOTE | Shangguan et al., arXiv 2024 | Own implementation (`k=5`) |
| Self-paced Ensemble | Liu et al., ICDE 2020 | Own implementation (10 members, 20 bins) |

## Cross-dataset experiments

- **Frozen:** models trained on the source training split, tested on the full target dataset.
- **Retrained:** models trained on the full source dataset, tested on the full target dataset.
- **Reference:** UNIDECOR models tested on UNIDECOR's own test split (for comparison).

## Results

Final tables for the paper are in `results/summary_tables/`:

- `in_domain_baseline.csv`: no rebalancing
- `in_domain_rebalancing.csv`: all techniques, with change versus baseline
- `cross_dataset.csv`: both directions, frozen and retrained
- `cross_dataset_frozen_per_domain.csv`: single-domain DiFrauD models tested on UNIDECOR

## Limitations

- Three UNIDECOR components (DEC_OP, BLT_C, ONLINE_DE) were not available.
- "Deceptive" means fraud/spam/fake news in DiFrauD but lying in UNIDECOR, which limits cross-dataset transfer.
- No hyperparameter or threshold tuning.
- Three of the four techniques are our own implementations of the cited papers.
