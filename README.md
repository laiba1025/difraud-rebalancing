# Rebalancing techniques and cross-dataset generalization: DiFrauD and UNIDECOR

This project studies the effect of four recent rebalancing techniques on three standard classifiers
on DiFrauD, per domain and pooled across domains. It also tests cross-dataset generalization
between DiFrauD and UNIDECOR in both directions, with frozen in-domain models and with models
retrained on the full source dataset.

## Setup

Commands are for Git Bash on Windows, from the project root. On Linux/macOS use `.venv/bin/python`.
Scripts are run from `src/` with `PYTHONUTF8=1`.

1. **Python environment** (Python 3.11.9):

   ```
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt
   ```

2. **DiFrauD:** downloaded automatically on first use (pinned HuggingFace commit, see "Data").

3. **UNIDECOR:** builds `unidecor_repo/datasets/consolidated_deception_data.json`:

   ```
   git clone -b unidecor_v1.0 https://bitbucket.org/aswathyve/unifieddeceptioncorpus.git unidecor_repo
   cd unidecor_repo
   ../.venv/Scripts/python download.py --yes          # --yes accepts all licence prompts
   # The three UMich downloads fail with an SSL error in download.py; fetch them with curl
   cd datasets/downloaded
   for p in cross_cultural_deception:crossCulturalDeception.2014.tar.gz \
            open_domain_deception:openDeception.2015.tar.gz \
            real_life_trial_data:RealLifeDeceptionDetection.2016.zip; do
     d=${p%%:*}; f=${p#*:}; mkdir -p $d
     curl -sS -f -L -o $d/$f "https://web.eecs.umich.edu/~mihalcea/downloads/$f"
     ../../../.venv/Scripts/python -c "import shutil,os,sys; shutil.unpack_archive(sys.argv[1], sys.argv[2]); os.remove(sys.argv[1])" $d/$f $d
   done
   cd ../..
   PYTHONUTF8=1 ../.venv/Scripts/python get_consolidated_data.py   # also downloads Diplomacy
   cd ..
   ```

`models/` and `cache/` are created by the scripts (both git-ignored).

## Data

### DiFrauD

- **Source:** HuggingFace `redasers/difraud`, pinned to commit `f8fa04aff083aa22c6878d005b4655e34a4cbd85`.
- **Domains:** 7, each with train/validation/test splits: fake_news, job_scams, phishing,
  political_statements, product_reviews, sms and twitter_rumours. 95,854 texts in total.
- **Labels:** `1` = deceptive, `0` = non-deceptive.
- **Loading:** recent `datasets` versions refuse to run the dataset's loading script, so the same
  `<domain>/<split>.jsonl` files are read directly with `huggingface_hub`. The data are identical.

### UNIDECOR

Built with the scripts at https://bitbucket.org/aswathyve/unifieddeceptioncorpus/ (tag
`unidecor_v1.0`). **11 of 14 components were obtained.**

| Component | Obtained | Note |
|---|---|---|
| OP_SPAM, DE_REV_2014, DE_REV_2018, BLUFF | yes | `download.py` |
| CROSS_CULT_DE, OPEN_DOMAIN, TRIAL_DE | yes | same URLs fetched with curl (SSL error in `download.py`; certificate verification kept on) |
| DIPLOMACY_CONVO | yes | via convokit |
| MU3D, BOX_OF_LIES | yes | files already included in the repo |
| MAFIASCUM | yes | latest version; the pinned commit could not be checked out |
| DEC_OP, BLT_C, ONLINE_DE | **no** | only available on request from the authors / LDC |

**Mapping to DiFrauD's format** (`load_unidecor()`):
- `truth_label` D → 1 (deceptive) and T → 0. Each component is treated as a domain.
- Non-English texts are removed (352, all in CROSS_CULT_DE), leaving **41,968 texts** (26.0%
  deceptive).
- Mafiascum is kept untruncated.
- **Splits** (for the frozen UNIDECOR models): DIPLOMACY_CONVO keeps its official split; every
  other component gets a stratified 80/10/10 split (seed 42). This gives train 32,871, validation
  3,884 and test 5,213, saved in `results/cross_dataset/unidecor_splits.csv`.

## Experimental setup

- **Positive class = deceptive (1) everywhere.** Metrics: macro-F1, precision/recall/F1 of the
  deceptive class, and AUC-ROC. All are reported as mean and std over seeds 42, 43 and 44.
- **Features:** TF-IDF (word 1-2-grams, sublinear TF, `min_df=2`, `max_features=50000`) followed by
  TruncatedSVD to 300 dimensions. They are fitted on **source training data only**, in every
  experiment, and are fixed across seeds.
- **Classifiers:** `LogisticRegression`, `LinearSVC` and `XGBClassifier` (`tree_method="hist"`),
  with default hyperparameters. The baseline is therefore identical across seeds (std 0).
- **Threshold:** the default (probability 0.5; SVM decision value 0) is used deliberately and not
  tuned. AUC-ROC is threshold-free.
- **In-domain:** for each DiFrauD domain and for "pooled" (all domains), train on the train split
  and test on the test split. **The validation split is not used** (no tuning).
- **Rebalancing** is applied after the SVD, to training features only.
- **Minority class:**
  - In political_statements the training minority class is **truthful** (35.6%), so the
    oversamplers add truthful samples there.
  - product_reviews is balanced (50.03%), so the oversamplers add only 10 points.

## Rebalancing techniques

| Technique | Source | Implementation | Parameters |
|---|---|---|---|
| Simplicial SMOTE | O. Kachan, A. Savchenko, G. Gusev. *Simplicial SMOTE: Oversampling Solution to the Imbalanced Learning Problem.* KDD 2025. [arXiv:2503.03418](https://arxiv.org/abs/2503.03418) | **Our own implementation** (no official code available) | `k=10`, max simplex dimension `p=3` (the paper's runtime setting; the paper gives no default) |
| Multivariate Gaussian SMOTE (MGS) | A. Sakho, E. Malherbe, E. Scornet. *Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants.* AISTATS 2026. [arXiv:2402.03819](https://arxiv.org/abs/2402.03819) | Algorithm of the authors' package [`mgs-grf`](https://github.com/artefactory/mgs-grf) 0.1.0, run in batches to fit in memory (output verified identical) | package defaults: `K` = number of features = **300** |
| MEB-SMOTE | Y.-Y. Shangguan, S.-S. Chen, X.-Y. Li. *Minimum Enclosing Ball Synthetic Minority Oversampling Technique from a Geometric Perspective.* **arXiv preprint** arXiv:2408.03526, 2024. [arXiv:2408.03526](https://arxiv.org/abs/2408.03526) | **Our own implementation** (no code available) | `k=5` (as in the paper) |
| Self-paced Ensemble (SPE) | Z. Liu, W. Cao, Z. Gao, J. Bian, H. Chen, Y. Chang, T.-Y. Liu. *Self-paced Ensemble for Highly Imbalanced Massive Data Classification.* ICDE 2020. [arXiv:1909.03500](https://arxiv.org/abs/1909.03500) | **Our own implementation** (official packages incompatible with current scikit-learn) | `n=10` members, `k=20` hardness bins (the paper's settings) |

SPE is from 2020. It is included because the NeurIPS 2025 CLIMB benchmark
([arXiv:2505.17451](https://arxiv.org/abs/2505.17451)) ranks it best of 29 rebalancing methods.

**Implementation notes** (details in `src/rebalancing.py`; checks in `results/rebalancing_checks/`):
- Oversamplers add synthetic minority points until both classes are equal in size.
- **MGS, K = 300:** with few minority points (e.g. 479 in job_scams), each Gaussian covers most of
  the minority class, so synthetic points fall outside the minority region more often. The default
  is kept deliberately.
- **MGS neighbour search:** the in-domain runs used `ball_tree` (as in `mgs-grf`); the
  cross-dataset runs used the faster, also exact, `brute`. Both reproduce the package's output on
  job_scams (max difference 1.5e-8).
- **MEB-SMOTE:** the neighbour set excludes the selected point; the minimum enclosing ball is
  computed exactly.
- **SPE:**
  - For LinearSVC, the member score is sigmoid(`decision_function`).
  - If a hardness bin is too small for its share, the shortfall is passed to the other bins, so
    each member trains on exactly |P| minority + |P| majority samples. The paper does not cover
    this case.
- **Sanity checks:** class counts after resampling, minority-region checks, a toy 2-D plot, a
  comparison with imbalanced-learn's SMOTE, and equivalence with `mgs-grf`.

## Cross-dataset experiments

Both directions, DiFrauD → UNIDECOR and UNIDECOR → DiFrauD, with the same techniques, classifiers
and seeds. All fitting (TF-IDF, SVD, resampling) uses source data only.

- **Frozen:** models trained on the source's own training split are applied to the full target
  without refitting.
  - DiFrauD → UNIDECOR uses the saved in-domain models: *pooled* (main result) and each of the 7
    per-domain models.
  - UNIDECOR → DiFrauD uses models trained on the UNIDECOR training split.
- **Retrained:** features and models are fitted on *all* of the source dataset, then tested on the
  full target.
- **In-domain reference:** the frozen UNIDECOR models on UNIDECOR's own test split. Labelled as a
  reference, not a cross-dataset result.
- **Reporting:** whole target (`ALL`) and per DiFrauD domain / UNIDECOR component.
- **`low_n` rule:** groups with fewer than 500 texts, or fewer than 150 in either class, are
  flagged. On the full UNIDECOR target these are TRIAL_DE, DE_REV_2014, MU3D and BOX_OF_LIES.

## How to run (order)

From `src/`:

| # | Command | Output |
|---|---|---|
| 1 | `PYTHONUTF8=1 ../.venv/Scripts/python inspect_data.py` | `results/data_stats/` |
| 2 | `PYTHONUTF8=1 ../.venv/Scripts/python check_cross_duplicates.py` | `results/data_stats/` |
| 3 | `PYTHONUTF8=1 ../.venv/Scripts/python run_in_domain.py --techniques baseline` | `results/in_domain/`, `models/in_domain/` |
| 4 | `PYTHONUTF8=1 ../.venv/Scripts/python check_rebalancing.py` | `results/rebalancing_checks/` |
| 5 | `PYTHONUTF8=1 ../.venv/Scripts/python run_in_domain.py --techniques simplicial mgs meb spe --reuse-features --out results/in_domain_rebalancing --baseline results/in_domain/summary.csv` | `results/in_domain_rebalancing/` |
| 6 | `PYTHONUTF8=1 ../.venv/Scripts/python run_cross_dataset.py` | `results/cross_dataset/`, `models/cross_dataset/` |
| 7 | `PYTHONUTF8=1 ../.venv/Scripts/python make_summary_tables.py` | `results/summary_tables/` |

The cross-dataset run (6) needs the saved models from 3 and 5. It runs in blocks that are saved as
they finish, so an interrupted run resumes where it stopped. The `--subsample N` options of
`run_in_domain.py` and `run_cross_dataset.py` are for quick tests only; their output goes to
`cache/`.

## Output files

### `results/summary_tables/` (final tables)
Each table has macro-F1, deceptive precision/recall/F1 and AUC-ROC, as mean and std over seeds.
- `in_domain_baseline.csv`: no rebalancing, per DiFrauD domain and pooled.
- `in_domain_rebalancing.csv`: all techniques, per domain and pooled, with the change versus the
  baseline.
- `cross_dataset.csv`: pooled source models, frozen and retrained, both directions, whole target
  and per group (`low_n` flagged), with the change versus the baseline. Reference rows are
  labelled `in-domain reference (not cross-dataset)`.
- `cross_dataset_frozen_per_domain.csv`: frozen per-domain DiFrauD models → UNIDECOR.

### Detailed results
- `results/data_stats/`: dataset statistics per domain/component, obtained UNIDECOR components,
  languages, and exact duplicates shared by the two datasets.
- `results/in_domain/`, `results/in_domain_rebalancing/`, `results/cross_dataset/`:
  - `runs.csv`: one row per run and seed.
  - `summary.csv`: mean and std over seeds.
  - `delta_vs_baseline.csv`: change versus the baseline.
  - `config.json`: settings and versions.
  - The cross-dataset folder also holds `unidecor_splits.csv` and `blocks/`.
- `results/rebalancing_checks/`: sanity and equivalence checks of the rebalancing implementations,
  and the toy plot `toy_2d_oversampling.png`.

## Assumptions and limitations

- **Different notions of "deceptive":** DiFrauD covers fraud, scams, spam, phishing, fake news and
  rumours. UNIDECOR covers statements known to be lies (experiments, games, trials). Cross-dataset
  results should be read with this in mind.
- **Partial UNIDECOR:** DEC_OP, BLT_C and ONLINE_DE are missing, and MAFIASCUM may differ from the
  version in the UNIDECOR paper.
- **Duplicate texts:** 6 texts appear in both datasets (6 DiFrauD rows, 28 UNIDECOR rows). Five
  are trivial ("Ok", "Thanks!", ...), and one has opposite labels in the two datasets. Given the
  tiny overlap, they are not removed.
- **Whole-target AUC (cross-dataset):** per-component AUCs are mostly lower than the whole-target
  AUC, so part of it can reflect differences between components rather than within them.
- **No tuning:** default hyperparameters and threshold. For example, SPE's balanced members shift
  predictions toward the deceptive class at threshold 0.5, which lowers macro-F1 on job_scams
  although AUC is unchanged.
- **Own implementations:** Simplicial SMOTE, MEB-SMOTE and SPE are our own implementations of the
  cited papers, checked in `results/rebalancing_checks/`. An SPE sampling bug (members got too few
  majority samples) was found and fixed before the experiments.
- **Run history:** the cross-dataset run was interrupted by memory pressure and resumed after a
  memory-only code change. Finished blocks were kept unchanged (checksum-verified), and results
  are unaffected.
- **Timings** (`*_seconds` columns) are not comparable across runs: the laptop was at times under
  memory pressure or on battery.

## Environment and pinned versions

Windows 11, Python 3.11.9 (`requirements.txt`):

| Package | Version | Package | Version |
|---|---|---|---|
| numpy | 2.4.6 | datasets | 5.1.0 |
| scipy | 1.17.1 | huggingface_hub | 1.33.0 |
| pandas | 3.0.6 | convokit | 4.1.2 |
| scikit-learn | 1.9.1 | openpyxl | 3.1.5 |
| xgboost | 3.2.0 | requests | 2.34.2 |
| imbalanced-learn | 0.14.2 | click | 8.5.0 |
| mgs-grf | 0.1.0 | matplotlib | 3.11.2 |
| networkx | 3.6.1 | joblib | 1.6.0 |
