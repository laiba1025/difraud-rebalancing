"""Cross-dataset experiments: DiFrauD -> UNIDECOR and UNIDECOR -> DiFrauD.
"""
import argparse
import gc
import json
import sys
import time
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd

from classifiers import CLASSIFIERS, make_classifier, scores
from data_loading import (DIFRAUD_DOMAINS, DIFRAUD_REVISION, ROOT, load_difraud,
                          load_unidecor, unidecor_splits)
from features import FEATURE_PARAMS, fit_features
from metrics import METRICS, evaluate
from rebalancing import TECHNIQUES, fit_model, resample

SEEDS = [42, 43, 44]
OUT = ROOT / "results" / "cross_dataset"
BLOCKS_DIR = OUT / "blocks"
MODELS = ROOT / "models"
IN_DOMAIN_MODELS = MODELS / "in_domain"         # frozen DiFrauD models (run_in_domain.py)
CROSS_MODELS = MODELS / "cross_dataset"
SUBSAMPLE = None    # quick test only (--subsample): stratified subsample of every source/target


def subsample(df):
    if not SUBSAMPLE or len(df) <= SUBSAMPLE:
        return df
    return df.groupby("label", group_keys=False).sample(frac=SUBSAMPLE / len(df), random_state=0)


def load_dif():
    return subsample(load_difraud())
KEYS = ["direction", "mode", "source_model", "technique", "classifier", "target_group"]


def low_n(n, n_dec):
    return bool(n < 500 or min(n_dec, n - n_dec) < 150)


def evaluate_groups(model, X, y, groups):
    """Metrics on the whole target ("ALL") and per group (domain / component)."""
    pred, score = model.predict(X), scores(model, X)
    rows = []
    for name, mask in [("ALL", np.ones(len(y), bool))] + [(g, groups == g) for g in sorted(set(groups))]:
        yt = y[mask]
        m = (evaluate(yt, pred[mask], score[mask]) if 0 < yt.sum() < len(yt)
             else {k: np.nan for k in METRICS})
        rows.append({"target_group": name, "n": int(mask.sum()), "n_deceptive": int(yt.sum()),
                     "pct_deceptive": round(100 * yt.mean(), 2),
                     "low_n": low_n(int(mask.sum()), int(yt.sum())), **m})
    return rows


def train_and_evaluate(X_tr, y_tr, targets, model_dir, info):
    """Train baseline + 4 techniques x 3 classifiers x 3 seeds on (X_tr, y_tr) and evaluate on
    every target. targets: list of (meta dict, X, y, groups)."""
    rows = []
    for tech in TECHNIQUES:
        for seed in SEEDS:
            t0 = time.time()
            # MGS uses the brute-force neighbour search in the cross-dataset runs.
            X_fit, y_fit = resample("mgs_brute" if tech == "mgs" else tech, X_tr, y_tr, seed)
            resample_s = time.time() - t0
            for clf_name in CLASSIFIERS:
                t1 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model, n_fit, after = fit_model(tech, make_classifier(clf_name, seed),
                                                    X_fit, y_fit, seed)
                fit_s = time.time() - t1
                (model_dir / tech).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_dir / tech / f"{clf_name}_seed{seed}.joblib")
                train_info = {**info, "n_train_fit": n_fit,
                              "n_truthful_after": int(after[0]), "n_deceptive_after": int(after[1]),
                              "resample_seconds": round(resample_s, 2),
                              "fit_seconds": round(fit_s, 2)}
                first = len(rows)
                for meta, X, y, groups in targets:
                    for r in evaluate_groups(model, X, y, groups):
                        rows.append({**meta, "technique": tech, "classifier": clf_name,
                                     "seed": seed, **r, **train_info})
                all_row = rows[first]                # "ALL" of the first target
                print(f"  {tech:10s} {clf_name:10s} seed={seed} ALL macroF1={all_row['macro_f1']:.4f} "
                      f"AUC={all_row['auc_roc']:.4f} (resample {resample_s:.0f}s, fit {fit_s:.0f}s)",
                      flush=True)
    return rows


def load_uni():
    uni = load_unidecor()
    uni["split"] = unidecor_splits(uni)
    return uni


def load_uni_sub(split=None):
    uni = load_uni()
    return subsample(uni if split is None else uni[uni["split"] == split])


def frozen_difraud(setting):
    """Saved in-domain DiFrauD models (trained on the DiFrauD train split) -> full UNIDECOR."""
    pipe = joblib.load(IN_DOMAIN_MODELS / setting / "features.joblib")
    uni = load_uni_sub()
    X = pipe.transform(uni["text"]).astype("float32")
    y, groups = uni["label"].to_numpy(), uni["domain"].to_numpy()
    del uni                                   # raw texts are not needed after featurising
    gc.collect()
    meta = {"direction": "DiFrauD->UNIDECOR", "mode": "frozen", "source_model": setting,
            "source_train_data": f"DiFrauD {setting} train split"}
    rows = []
    for tech in TECHNIQUES:
        for seed in SEEDS:
            for clf_name in CLASSIFIERS:
                model = joblib.load(IN_DOMAIN_MODELS / setting / tech / f"{clf_name}_seed{seed}.joblib")
                for r in evaluate_groups(model, X, y, groups):
                    rows.append({**meta, "technique": tech, "classifier": clf_name, "seed": seed, **r})
        print(f"  {tech} done", flush=True)
    return rows


def train_source(name, source_fn, targets_fn, model_dir, meta_info):
    """Fit TF-IDF + SVD on the source texts only, transform the targets, train and evaluate.

    source_fn / targets_fn load the data inside the block; raw texts are dropped as soon as they
    are featurised, and the feature matrices are freed when the block ends (memory only; this
    does not change any result)."""
    t = time.time()
    src = source_fn()
    y_tr = src["label"].to_numpy()
    pipe, X_tr = fit_features(src["text"])
    del src
    gc.collect()
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "features.joblib")
    targets = []
    for meta, fn in targets_fn:
        df = fn()
        targets.append((meta, pipe.transform(df["text"]).astype("float32"),
                        df["label"].to_numpy(), df["domain"].to_numpy()))
        del df
        gc.collect()
    del pipe
    print(f"[{name}] source n={len(y_tr)} %deceptive={100 * y_tr.mean():.1f} "
          f"features+target transform {time.time() - t:.0f}s", flush=True)
    info = {**meta_info, "n_train": len(y_tr), "train_pct_deceptive": round(100 * y_tr.mean(), 2)}
    rows = train_and_evaluate(X_tr, y_tr, targets, model_dir, info)
    del X_tr, targets
    gc.collect()
    return rows


def summarise(runs):
    agg = {"n": ("n", "first"), "n_deceptive": ("n_deceptive", "first"),
           "pct_deceptive": ("pct_deceptive", "first"), "low_n": ("low_n", "first"),
           "n_seeds": ("seed", "nunique"),
           **{f"{k}_{s}": (k, s) for k in METRICS for s in ["mean", "std"]}}
    return runs.groupby(KEYS, sort=False).agg(**agg).reset_index()


def delta_vs_baseline(summary):
    base_keys = [k for k in KEYS if k != "technique"]
    base = summary[summary["technique"] == "baseline"][base_keys + [f"{m}_mean" for m in METRICS]]
    d = summary[summary["technique"] != "baseline"].merge(base, on=base_keys, suffixes=("", "_baseline"))
    for m in METRICS:
        d[f"delta_{m}"] = d[f"{m}_mean"] - d[f"{m}_mean_baseline"]
    cols = KEYS + ["n", "low_n"]
    for m in METRICS:
        cols += [f"{m}_mean_baseline", f"{m}_mean", f"delta_{m}"]
    return d[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=None, help="run only these blocks")
    ap.add_argument("--subsample", type=int, default=None,
                    help="quick test: stratified subsample of N rows of every source and target; "
                         "uses the in-domain quick-test models and writes to cache/ and models/*_subsampleN")
    args = ap.parse_args()
    global OUT, BLOCKS_DIR, IN_DOMAIN_MODELS, CROSS_MODELS, SUBSAMPLE
    if args.subsample:
        SUBSAMPLE = args.subsample
        OUT = ROOT / "cache" / f"cross_dataset_subsample{SUBSAMPLE}"
        BLOCKS_DIR = OUT / "blocks"
        IN_DOMAIN_MODELS = MODELS / f"in_domain_subsample{SUBSAMPLE}"
        CROSS_MODELS = MODELS / f"cross_dataset_subsample{SUBSAMPLE}"
    OUT.mkdir(parents=True, exist_ok=True)
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    # Data are loaded inside each block (not kept for the whole run) to limit memory use.
    uni = load_uni()
    uni[["text_ID", "domain", "label", "split"]].rename(columns={"domain": "component"}).to_csv(
        OUT / "unidecor_splits.csv", index=False)
    del uni
    gc.collect()
    uni_train = lambda: load_uni_sub("train")
    uni_test = lambda: load_uni_sub("test")

    blocks = {}
    for setting in ["pooled"] + DIFRAUD_DOMAINS:
        blocks[f"frozen_difraud_{setting}"] = lambda s=setting: frozen_difraud(s)
    blocks["frozen_unidecor"] = lambda: train_source(
        "UNIDECOR train split", uni_train,
        [({"direction": "UNIDECOR->DiFrauD", "mode": "frozen", "source_model": "pooled"}, load_dif),
         ({"direction": "UNIDECOR->UNIDECOR", "mode": "in_domain_reference", "source_model": "pooled"},
          uni_test)],
        CROSS_MODELS / "unidecor_train_split", {"source_train_data": "UNIDECOR train split"})
    blocks["retrained_unidecor"] = lambda: train_source(
        "UNIDECOR all", load_uni_sub,
        [({"direction": "UNIDECOR->DiFrauD", "mode": "retrained", "source_model": "pooled"}, load_dif)],
        CROSS_MODELS / "unidecor_all", {"source_train_data": "UNIDECOR all"})
    blocks["retrained_difraud"] = lambda: train_source(
        "DiFrauD all", load_dif,
        [({"direction": "DiFrauD->UNIDECOR", "mode": "retrained", "source_model": "pooled"}, load_uni_sub)],
        CROSS_MODELS / "difraud_all", {"source_train_data": "DiFrauD all splits"})

    for name, fn in blocks.items():
        path = BLOCKS_DIR / f"{name}.csv"
        if (args.blocks and name not in args.blocks) or path.exists():
            if path.exists():
                print(f"== block {name} already done, skipped", flush=True)
            continue
        print(f"== block {name}", flush=True)
        t = time.time()
        pd.DataFrame(fn()).to_csv(path, index=False)
        print(f"== block {name} saved ({time.time() - t:.0f}s)", flush=True)
        gc.collect()

    done = [BLOCKS_DIR / f"{n}.csv" for n in blocks if (BLOCKS_DIR / f"{n}.csv").exists()]
    runs = pd.concat([pd.read_csv(p) for p in done], ignore_index=True)
    runs.to_csv(OUT / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    delta_vs_baseline(summary).to_csv(OUT / "delta_vs_baseline.csv", index=False, float_format="%.4f")
    config = {"difraud_revision": DIFRAUD_REVISION, "seeds": SEEDS, "techniques": TECHNIQUES,
              "subsample": SUBSAMPLE,
              "classifiers": {c: repr(make_classifier(c, 0)) for c in CLASSIFIERS},
              "features": FEATURE_PARAMS, "mgs_knn_algorithm": "brute",
              "low_n_rule": "n < 500 or fewer than 150 texts in either class",
              "blocks_done": [p.stem for p in done], "blocks_all": list(blocks),
              "python": sys.version.split()[0],
              "versions": {p: version(p) for p in ["scikit-learn", "xgboost", "numpy", "pandas",
                                                   "mgs-grf", "networkx"]}}
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved to {OUT} ({len(done)}/{len(blocks)} blocks)", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
