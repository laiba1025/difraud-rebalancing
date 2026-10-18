"""In-domain experiments on DiFrauD: per domain and pooled across all domains.
"""
import argparse
import json
import sys
import time
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from classifiers import CLASSIFIERS, make_classifier, scores
from data_loading import DIFRAUD_DOMAINS, DIFRAUD_REVISION, ROOT, load_difraud
from features import FEATURE_PARAMS, fit_features
from metrics import METRICS, evaluate
from rebalancing import TECHNIQUES, fit_model, resample

SEEDS = [42, 43, 44]
SETTING_NOTES = {
    "political_statements": "minority class is truthful (35.6%); oversamplers add truthful samples",
    "product_reviews": "balanced (50.03% deceptive); oversamplers add only 10 truthful samples",
}
DELTA_METRICS = ["macro_f1", "f1_deceptive", "auc_roc"]


def stratified_subsample(df, n, seed=0):
    """Stratified subsample of at most n rows (quick test runs only)."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)


def minority_class(pct_deceptive):
    """Minority class of the training data; "balanced" if within 1 point of 50%."""
    if abs(pct_deceptive - 50) < 1:
        return "balanced"
    return "deceptive" if pct_deceptive < 50 else "truthful"


def settings(data):
    for dom in DIFRAUD_DOMAINS:
        yield dom, data[data["domain"] == dom]
    yield "pooled", data


def summarise(runs):
    return (runs.groupby(["setting", "technique", "classifier"], sort=False)
            .agg(train_minority_class=("train_minority_class", "first"),
                 n_seeds=("seed", "nunique"),
                 **{f"{k}_{s}": (k, s) for k in METRICS for s in ["mean", "std"]})
            .reset_index()
            .assign(setting_note=lambda d: d["setting"].map(SETTING_NOTES).fillna("")))


def delta_vs_baseline(summary, baseline):
    """Mean change vs the baseline (same setting and classifier) for DELTA_METRICS."""
    base = baseline[["setting", "classifier"] + [f"{m}_mean" for m in DELTA_METRICS]]
    d = summary.merge(base, on=["setting", "classifier"], suffixes=("", "_baseline"))
    cols = ["setting", "technique", "classifier", "train_minority_class"]
    for m in DELTA_METRICS:
        d[f"{m}_baseline"] = d[f"{m}_mean_baseline"]
        d[f"delta_{m}"] = d[f"{m}_mean"] - d[f"{m}_mean_baseline"]
        cols += [f"{m}_baseline", f"{m}_mean", f"delta_{m}"]
    return d[cols + ["setting_note"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--techniques", nargs="+", default=TECHNIQUES, choices=TECHNIQUES)
    ap.add_argument("--classifiers", nargs="+", default=CLASSIFIERS, choices=CLASSIFIERS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--settings", nargs="+", default=None, help="subset of settings to run")
    ap.add_argument("--subsample", type=int, default=None,
                    help="quick test: subsample train and test of each setting to N rows")
    ap.add_argument("--reuse-features", action="store_true",
                    help="load the feature pipeline saved by an earlier run instead of refitting")
    ap.add_argument("--baseline", default=None, help="baseline summary.csv for delta_vs_baseline.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"subsample{args.subsample}" if args.subsample else "full"
    out = ROOT / (args.out or ("results/in_domain" if tag == "full" else f"cache/in_domain_{tag}"))
    model_dir = ROOT / "models" / ("in_domain" if tag == "full" else f"in_domain_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    data = load_difraud()
    runs_path = out / "runs.csv"
    # Re-running with other techniques appends; rows for re-run combinations are replaced.
    old = pd.read_csv(runs_path) if runs_path.exists() else None
    rows = []

    for setting, df in settings(data):
        if args.settings and setting not in args.settings:
            continue
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        if args.subsample:
            train = stratified_subsample(train, args.subsample)
            test = stratified_subsample(test, args.subsample)
        t0 = time.time()
        feat_path = model_dir / setting / "features.joblib"
        if args.reuse_features and feat_path.exists():
            pipe = joblib.load(feat_path)
            X_train = pipe.transform(train["text"]).astype("float32")
        else:
            pipe, X_train = fit_features(train["text"])
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, feat_path)
        X_test = pipe.transform(test["text"]).astype("float32")
        y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
        feat_s = time.time() - t0
        pct_dec = 100 * y_train.mean()
        minority = minority_class(pct_dec)
        before = np.bincount(y_train, minlength=2)
        print(f"[{setting}] train={len(train)} test={len(test)} "
              f"train %deceptive={pct_dec:.1f} features {feat_s:.0f}s", flush=True)

        for tech in args.techniques:
            for seed in args.seeds:
                t1 = time.time()
                X_fit, y_fit = resample(tech, X_train, y_train, seed)
                resample_s = time.time() - t1
                for clf_name in args.classifiers:
                    t2 = time.time()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model, n_fit, after = fit_model(
                            tech, make_classifier(clf_name, seed), X_fit, y_fit, seed)
                    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
                    fit_s = time.time() - t2
                    m = evaluate(y_test, model.predict(X_test), scores(model, X_test))
                    (model_dir / setting / tech).mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_dir / setting / tech / f"{clf_name}_seed{seed}.joblib")
                    rows.append({"setting": setting, "technique": tech, "classifier": clf_name,
                                 "seed": seed, "n_train": len(y_train), "n_train_fit": n_fit,
                                 "n_test": len(y_test), "train_pct_deceptive": round(pct_dec, 2),
                                 "train_minority_class": minority, **m,
                                 "fit_seconds": round(fit_s, 2),
                                 "convergence_warnings": n_conv,
                                 "resample_seconds": round(resample_s, 2),
                                 "n_truthful_before": int(before[0]),
                                 "n_deceptive_before": int(before[1]),
                                 "n_truthful_after": int(after[0]),
                                 "n_deceptive_after": int(after[1]),
                                 "setting_note": SETTING_NOTES.get(setting, "")})
                    print(f"  {tech:10s} {clf_name:10s} seed={seed} macroF1={m['macro_f1']:.4f} "
                          f"AUC={m['auc_roc']:.4f} (resample {resample_s:.1f}s, fit {fit_s:.1f}s)",
                          flush=True)

    runs = pd.DataFrame(rows)
    if old is not None:
        key = ["setting", "technique", "classifier", "seed"]
        keep = old.merge(runs[key], on=key, how="left", indicator=True)["_merge"] == "left_only"
        runs = pd.concat([old[keep.to_numpy()], runs], ignore_index=True)
    runs.to_csv(runs_path, index=False)

    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    if args.baseline:
        delta = delta_vs_baseline(summary, pd.read_csv(ROOT / args.baseline))
        delta.to_csv(out / "delta_vs_baseline.csv", index=False, float_format="%.4f")

    config = {"difraud_revision": DIFRAUD_REVISION, "seeds": args.seeds,
              "techniques": args.techniques, "subsample": args.subsample,
              "reuse_features": args.reuse_features, "features": FEATURE_PARAMS,
              "classifiers": {c: repr(make_classifier(c, 0)) for c in CLASSIFIERS},
              "evaluation": "train split -> test split; validation unused; positive class = deceptive",
              "class_counts_after": "oversamplers: resampled training set; spe: first member's "
                                    "training subset (minority + equal-size majority sample)",
              "python": sys.version.split()[0],
              "versions": {p: version(p) for p in
                           ["scikit-learn", "xgboost", "numpy", "pandas", "mgs-grf", "networkx"]}}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
