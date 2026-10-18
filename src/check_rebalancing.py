"""Sanity checks for the rebalancing implementations.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.spatial import Delaunay
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from classifiers import CLASSIFIERS, make_classifier, scores
from data_loading import ROOT, load_difraud
from metrics import evaluate
from rebalancing import (MGS, MEBSMOTE, SelfPacedEnsemble, SimplicialSMOTE,
                         minimum_enclosing_ball)

OUT = ROOT / "results" / "rebalancing_checks"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
warnings.filterwarnings("ignore", message="MGSGRFOverSampler called")


def samplers(seed=SEED):
    return {"smote_imblearn": SMOTE(random_state=seed),
            "simplicial": SimplicialSMOTE(k=10, p=3, random_state=seed),
            "mgs": MGS(random_state=seed),
            "meb": MEBSMOTE(k=5, random_state=seed)}


def toy_data(n=1200, n_minority=60, seed=0):
    X, y = make_moons(n_samples=n, noise=0.15, random_state=seed)
    rng = np.random.RandomState(seed)
    keep = np.concatenate([np.where(y == 0)[0], rng.choice(np.where(y == 1)[0], n_minority, replace=False)])
    return X[keep], y[keep]


def region_checks(X, y, X_syn, minority, hull=False):
    Xm = X[y == minority]
    lo, hi = Xm.min(axis=0), Xm.max(axis=0)
    out = {"in_minority_bbox": float(np.all((X_syn >= lo - 1e-9) & (X_syn <= hi + 1e-9), axis=1).mean())}
    if hull:
        out["in_minority_hull"] = float((Delaunay(Xm).find_simplex(X_syn, tol=1e-9) >= 0).mean())
    idx = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(X_syn, return_distance=False)[:, 0]
    out["nn_is_minority"] = float((y[idx] == minority).mean())
    real_min = np.where(y == minority)[0]
    idx2 = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X[real_min], return_distance=False)
    other = np.where(idx2[:, 0] == real_min, idx2[:, 1], idx2[:, 0])
    out["ref_nn_is_minority"] = float((y[other] == minority).mean())
    nn_m = NearestNeighbors(n_neighbors=2).fit(Xm)
    d_syn = nn_m.kneighbors(X_syn, n_neighbors=1)[0][:, 0]
    d_ref = nn_m.kneighbors(Xm)[0][:, 1]
    out["dist_ratio"] = float(np.median(d_syn) / np.median(d_ref))
    return out


def counts(y):
    return {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}


def toy_checks():
    X, y = toy_data()
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4), sharex=True, sharey=True)
    axes[0].scatter(*X[y == 0].T, s=5, c="#9aa5b1", label="majority (0)")
    axes[0].scatter(*X[y == 1].T, s=14, c="#d1495b", label="minority (1)")
    axes[0].set_title(f"original {counts(y)}")
    axes[0].legend(loc="lower left", fontsize=8)
    for ax, (name, s) in zip(axes[1:], samplers().items()):
        Xr, yr = s.fit_resample(X, y)
        X_syn = Xr[len(X):]
        rows.append({"data": "toy_2d", "sampler": name, "counts_before": counts(y),
                     "counts_after": counts(yr), "n_synthetic": len(X_syn),
                     **region_checks(X, y, X_syn, 1, hull=True)})
        ax.scatter(*X[y == 0].T, s=5, c="#9aa5b1")
        ax.scatter(*X_syn.T, s=4, c="#edae49", alpha=0.6, label="synthetic")
        ax.scatter(*X[y == 1].T, s=14, c="#d1495b")
        ax.set_title(f"{name} {counts(yr)}")
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "toy_2d_oversampling.png", dpi=110)
    return rows


def meb_solver_check(n_sets=200, seed=SEED):
    rng = np.random.RandomState(seed)
    worst_out, worse_than_approx, worse_than_centroid = 0.0, 0, 0
    for t in range(n_sets):
        d = 2 if t % 2 else 300
        P = rng.normal(size=(5, d)) * rng.uniform(0.1, 3, size=d)
        c, r = minimum_enclosing_ball(P)
        worst_out = max(worst_out, float((np.linalg.norm(P - c, axis=1) - r).max()))
        # Badoiu-Clarkson iterative (1+eps)-approximation of the MEB.
        a = P.mean(axis=0)
        for i in range(1, 5001):
            far = P[np.argmax(np.linalg.norm(P - a, axis=1))]
            a = a + (far - a) / (i + 1)
        r_approx = np.linalg.norm(P - a, axis=1).max()
        r_centroid = np.linalg.norm(P - P.mean(axis=0), axis=1).max()
        worse_than_approx += r > r_approx + 1e-9
        worse_than_centroid += r > r_centroid + 1e-9
    return {"n_point_sets": n_sets, "points_per_set": 5, "dims": [2, 300],
            "max_point_outside_ball": worst_out,
            "n_radius_larger_than_badoiu_clarkson": int(worse_than_approx),
            "n_radius_larger_than_centroid_ball": int(worse_than_centroid)}


def job_scams_features():
    d = load_difraud()
    js = d[d["domain"] == "job_scams"]
    tr, te = js[js["split"] == "train"], js[js["split"] == "test"]
    path = ROOT / "models" / "in_domain" / "job_scams" / "features.joblib"
    if path.exists():
        pipe = joblib.load(path)
    else:  # e.g. a clean checkout: fit on the job_scams train split only (same settings)
        from features import fit_features
        pipe = fit_features(tr["text"])[0]
    return (pipe.transform(tr["text"]).astype("float32"), tr["label"].to_numpy(),
            pipe.transform(te["text"]).astype("float32"), te["label"].to_numpy())


def mgs_equivalence(X, y):
    from mgs_grf import MGSGRFOverSampler
    Xp, yp = MGSGRFOverSampler(categorical_features=None, random_state=SEED).fit_resample(X, y)
    Xo, yo = MGS(random_state=SEED).fit_resample(X, y)
    return {"data": "job_scams train features", "shape_package": list(Xp.shape),
            "shape_ours": list(Xo.shape), "labels_equal": bool(np.array_equal(yp, yo)),
            "max_abs_diff": float(np.abs(np.asarray(Xp, float) - np.asarray(Xo, float)).max()),
            "allclose_atol_1e-6": bool(np.allclose(np.asarray(Xp, float), np.asarray(Xo, float), atol=1e-6))}


def mgs_brute_equivalence(X, y):
    """Cross-dataset runs use MGS with brute-force neighbour search instead of ball_tree.
    Compare it with mgs-grf's output, and compare the neighbour lists of the two searches; any
    difference must come from exactly tied distances (e.g. duplicate texts)."""
    from mgs_grf import MGSGRFOverSampler
    Xp, yp = MGSGRFOverSampler(categorical_features=None, random_state=SEED).fit_resample(X, y)
    Xb, yb = MGS(random_state=SEED, knn_algorithm="brute").fit_resample(X, y)
    Xm = X[y == 1].astype(float)
    K = Xm.shape[1]
    nb_tree = (NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(Xm)
               .kneighbors(Xm, n_neighbors=K + 1, return_distance=False))
    nb_brute = (NearestNeighbors(n_neighbors=K, algorithm="brute").fit(Xm)
                .kneighbors(Xm, n_neighbors=K + 1, return_distance=False))
    diff_rows = [i for i in range(len(Xm)) if set(nb_tree[i]) != set(nb_brute[i])]
    # For rows whose neighbour sets differ, check the swapped points are exactly as far away.
    ties_only = True
    for i in diff_rows:
        a = set(nb_tree[i]) - set(nb_brute[i])
        b = set(nb_brute[i]) - set(nb_tree[i])
        da = np.sort(np.linalg.norm(Xm[list(a)] - Xm[i], axis=1))
        db = np.sort(np.linalg.norm(Xm[list(b)] - Xm[i], axis=1))
        ties_only &= bool(np.allclose(da, db, rtol=0, atol=1e-9))
    return {"data": "job_scams train features",
            "minority_points": int(len(Xm)), "K": int(K),
            "rows_with_different_neighbour_sets_ball_tree_vs_brute": len(diff_rows),
            "rows_with_different_neighbour_order": int(sum(not np.array_equal(nb_tree[i], nb_brute[i])
                                                           for i in range(len(Xm)))),
            "differences_are_exact_distance_ties_only": bool(ties_only),
            "shape_package": list(Xp.shape), "shape_brute": list(Xb.shape),
            "labels_equal": bool(np.array_equal(yp, yb)),
            "max_abs_diff_vs_package": float(np.abs(np.asarray(Xp, float) - np.asarray(Xb, float)).max()),
            "allclose_atol_1e-6": bool(np.allclose(np.asarray(Xp, float), np.asarray(Xb, float), atol=1e-6))}


def job_scams_comparison(X, y, Xte, yte):
    region_rows, score_rows = [], []
    for name, s in [("none", None), *samplers().items()]:
        t = time.time()
        Xr, yr = (X, y) if s is None else s.fit_resample(X, y)
        secs = time.time() - t
        if s is not None:
            row = {"data": "job_scams", "sampler": name, "counts_before": counts(y),
                   "counts_after": counts(yr), "n_synthetic": len(yr) - len(y),
                   "resample_seconds": round(secs, 1),
                   **region_checks(X, y, Xr[len(X):], 1)}
            if isinstance(s, SimplicialSMOTE):
                row["simplicial_n_maximal_simplices"] = s.n_simplices_
                row["simplicial_vertices_per_simplex"] = {int(k): int(v) for k, v in enumerate(s.simplex_sizes_) if v}
            region_rows.append(row)
        for clf_name in CLASSIFIERS:
            clf = make_classifier(clf_name, SEED).fit(Xr, yr)
            m = evaluate(yte, clf.predict(Xte), scores(clf, Xte))
            score_rows.append({"sampler": name, "classifier": clf_name, "n_train": len(yr),
                               "n_deceptive_train": int(yr.sum()), **{k: round(v, 4) for k, v in m.items()}})
            print(f"  {name:15s} {clf_name:10s} macroF1={m['macro_f1']:.4f} AUC={m['auc_roc']:.4f}", flush=True)
    return region_rows, score_rows


def spe_toy():
    X, y = toy_data(seed=1)
    Xte, yte = toy_data(seed=2)
    spe = SelfPacedEnsemble(LinearSVC(random_state=SEED), n_estimators=10, k_bins=20,
                            random_state=SEED).fit(X, y)
    dec = np.array([m.decision_function(Xte[:5]) for m in spe.estimators_])     # (10, 5)
    sig = 1 / (1 + np.exp(-dec))
    proba = spe.predict_proba(Xte)[:, 1]
    plain = LinearSVC(random_state=SEED).fit(X, y)
    out = {
        "train_counts": counts(y), "test_counts": counts(yte),
        "n_members": len(spe.estimators_),
        "member_training_subset_sizes": [int(s) for s in spe.subset_sizes_],
        "member_score_rule": "sigmoid(decision_function) per LinearSVC member; ensemble score = mean over members",
        "first5_test_points": {
            "member_decision_function": dec.round(4).tolist(),
            "member_sigmoid": sig.round(4).tolist(),
            "mean_of_member_sigmoids": sig.mean(axis=0).round(4).tolist(),
            "spe_predict_proba[:,1]": proba[:5].round(4).tolist(),
        },
        "predict_proba_equals_mean_sigmoid": bool(np.allclose(sig.mean(axis=0), proba[:5])),
        "predict_equals_score>=0.5": bool(np.array_equal(spe.predict(Xte), (proba >= 0.5).astype(int))),
        "auc_spe_linear_svm": round(roc_auc_score(yte, proba), 4),
        "auc_plain_linear_svm_decision_function": round(roc_auc_score(yte, plain.decision_function(Xte)), 4),
        "auc_spe_logreg": None, "auc_spe_xgboost": None,
    }
    for c in ["logreg", "xgboost"]:
        m = SelfPacedEnsemble(make_classifier(c, SEED), random_state=SEED).fit(X, y)
        out[f"auc_spe_{c}"] = round(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), 4)
    return out


if __name__ == "__main__":
    import sys
    if "--mgs-brute-only" in sys.argv:
        X, y, _, _ = job_scams_features()
        res = mgs_brute_equivalence(X, y)
        (OUT / "mgs_brute_equivalence.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        sys.exit(0)
    print("toy 2-D checks ...", flush=True)
    toy_rows = toy_checks()
    print("MEB solver check ...", flush=True)
    meb = meb_solver_check()
    print("job_scams features ...", flush=True)
    X, y, Xte, yte = job_scams_features()
    print("MGS equivalence ...", flush=True)
    mgs_eq = mgs_equivalence(X, y)
    print("job_scams comparison ...", flush=True)
    js_regions, js_scores = job_scams_comparison(X, y, Xte, yte)
    print("SPE toy ...", flush=True)
    spe = spe_toy()

    pd.DataFrame(toy_rows + js_regions).to_csv(OUT / "oversampling_region_checks.csv", index=False)
    pd.DataFrame(js_scores).to_csv(OUT / "job_scams_vs_smote_scores.csv", index=False)
    (OUT / "meb_solver_check.json").write_text(json.dumps(meb, indent=2))
    (OUT / "mgs_equivalence.json").write_text(json.dumps(mgs_eq, indent=2))
    (OUT / "spe_toy_scores.json").write_text(json.dumps(spe, indent=2))
    print("saved to", OUT)
