"""Rebalancing techniques, applied to the (SVD) training features only, never to test data.
"""
from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

TECHNIQUES = ["baseline", "simplicial", "mgs", "meb", "spe"]


def _minority_split(y):
    """Return (minority label, majority label, number of synthetic points to reach balance)."""
    labels, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)
    return labels[order[0]], labels[order[-1]], int(counts[order[-1]] - counts[order[0]])


def _neighbours_without_self(X, k):
    """Indices of the k nearest neighbours of each row of X within X, excluding the row itself."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    out = np.empty((len(X), k), dtype=int)
    for i, row in enumerate(idx):
        # Usually the point itself comes first; with exact duplicates it may not.
        row = row[row != i]
        out[i] = row[:k]
    return out


def _append(X, y, X_new, label):
    return (np.vstack([X, X_new.astype(X.dtype)]),
            np.concatenate([y, np.full(len(X_new), label, dtype=y.dtype)]))


# --------------------------------------------------------------------------------------------
# Simplicial SMOTE (Kachan, Savchenko, Gusev, KDD 2025, arXiv:2503.03418), Algorithm 1:
#   1. build the symmetric k-NN graph of the minority class (edge i-j if j is among the k nearest
#      neighbours of i or i among those of j);
#   2. take the maximal simplices of the p-skeleton of its clique complex: maximal cliques with
#      at most p+1 vertices, and every (p+1)-vertex subset of larger maximal cliques;
#   3. draw m = n_majority - n_minority simplices uniformly with replacement;
#   4. for each, draw barycentric coordinates lambda ~ Dirichlet(1, ..., 1) and output lambda^T X.
# Parameters: k=10, p=3 (no tuning; the setting of the paper's runtime experiment).
# --------------------------------------------------------------------------------------------
class SimplicialSMOTE:
    def __init__(self, k=10, p=3, random_state=None):
        self.k, self.p, self.random_state = k, p, random_state

    def maximal_simplices(self, Xm):
        nbrs = _neighbours_without_self(Xm, self.k)
        G = nx.Graph()
        G.add_nodes_from(range(len(Xm)))
        G.add_edges_from((i, int(j)) for i in range(len(Xm)) for j in nbrs[i])
        simplices = set()
        for clique in nx.find_cliques(G):
            clique = tuple(sorted(clique))
            if len(clique) <= self.p + 1:
                simplices.add(clique)
            else:
                simplices.update(combinations(clique, self.p + 1))
        return sorted(simplices)

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        minority, _, m = _minority_split(y)
        if m == 0:
            return X, y
        Xm = X[y == minority]
        simplices = self.maximal_simplices(Xm)
        self.n_simplices_ = len(simplices)
        self.simplex_sizes_ = np.bincount([len(s) for s in simplices])
        chosen = rng.randint(len(simplices), size=m)
        X_new = np.empty((m, X.shape[1]))
        for r, s in enumerate(chosen):
            verts = simplices[s]
            lam = rng.dirichlet(np.ones(len(verts)))
            X_new[r] = lam @ Xm[list(verts)]
        return _append(X, y, X_new, minority)


# --------------------------------------------------------------------------------------------
# MGS (Sakho, Malherbe, Scornet, AISTATS 2026, arXiv:2402.03819) as implemented in the
# authors' package mgs-grf 0.1.0, MGSGRFOverSampler with continuous features only and default
# settings: K = number of features, mean-centred (mu = mean of the point and its K neighbours),
# empirical covariance of those K+1 points, Cholesky sampling with 1e-8 jitter.
# The package builds a d x d covariance for EVERY minority point at once (n_min x 300 x 300
# float64 = 5 GB for fake_news, 21 GB pooled), which does not fit in 16 GB RAM. This class runs
# the same algorithm with the same random draws (same seed, same order: randint then normal)
# but only for the sampled centre points, in batches. check_rebalancing.py verifies that its
# output equals mgs-grf's output on job_scams, where the package fits in memory.
# --------------------------------------------------------------------------------------------
class MGS:
    """knn_algorithm: "ball_tree" as in mgs-grf (used for the in-domain runs), or "brute"
    (exact as well, much faster in 300-d; used for the cross-dataset runs). Both return the
    same neighbours except for exactly tied distances (verified in check_rebalancing.py)."""

    def __init__(self, K=None, random_state=None, batch=64, knn_algorithm="ball_tree"):
        self.K, self.random_state, self.batch = K, random_state, batch
        self.knn_algorithm = knn_algorithm

    def fit_resample(self, X, y):
        minority, _, m = _minority_split(y)
        if m == 0:
            return X, y
        Xm = X[y == minority].astype(float)
        n, d = Xm.shape
        K = self.K or d
        # Same neighbour search as the package (ball_tree, K+1 neighbours including the point).
        nbrs = (NearestNeighbors(n_neighbors=K, algorithm=self.knn_algorithm, n_jobs=-1).fit(Xm)
                .kneighbors(Xm, n_neighbors=K + 1, return_distance=False))
        rng = np.random.RandomState(self.random_state)  # == np.random.seed(random_state)
        centres = rng.randint(n, size=m)
        u = rng.normal(loc=0, scale=1, size=(m, d))
        X_new = np.empty((m, d))
        uniq = np.unique(centres)
        for b in range(0, len(uniq), self.batch):
            ids = uniq[b:b + self.batch]
            pts = Xm[nbrs[ids]]                                   # (b, K+1, d)
            mus = pts.sum(axis=1) / (K + 1)
            cen = pts - mus[:, None, :]
            covs = np.matmul(np.swapaxes(cen, 1, 2), cen) / (K + 1)
            As = np.linalg.cholesky(covs + 1e-8 * np.identity(d))
            for j, c in enumerate(ids):
                rows = np.where(centres == c)[0]
                X_new[rows] = mus[j] + u[rows] @ As[j].T
        return _append(X, y, X_new, minority)


# --------------------------------------------------------------------------------------------
# MEB-SMOTE (Shangguan, Chen, Li, arXiv:2408.03526, 2024), Algorithm 1:
#   N_new = N_maj - N_min; randomly select N_new minority samples (without replacement when
#   N_new <= N_min, otherwise with replacement - the paper does not say); for each selected x_i,
#   take its k nearest minority neighbours X_i^k (k=5, x_i itself not included - the paper's
#   wording "the k nearest neighbors of x_i form the set"), compute the minimum enclosing ball
#   MEB(c_i, r_i) of X_i^k, and output x_i + delta (c_i - x_i), delta ~ U(0, 1).
# The paper solves the MEB with Welzl's algorithm. The MEB is unique, so we compute the same
# ball exactly by enumerating candidate support sets (all subsets of the k points): the MEB is
# the smallest circumscribed ball of a subset that contains all k points.
# --------------------------------------------------------------------------------------------
def _circumcentre(Q):
    """Centre of the smallest ball with all rows of Q on its boundary (within their affine hull)."""
    q0 = Q[0]
    if len(Q) == 1:
        return q0
    A = Q[1:] - q0
    a = np.linalg.lstsq(2 * A @ A.T, (A * A).sum(axis=1), rcond=None)[0]
    return q0 + a @ A


def minimum_enclosing_ball(P, tol=1e-9):
    """Exact minimum enclosing ball of a small point set P (m x d). Returns (centre, radius)."""
    best_c, best_r = None, np.inf
    m = len(P)
    for size in range(1, min(m, P.shape[1] + 1) + 1):
        for S in combinations(range(m), size):
            c = _circumcentre(P[list(S)])
            r = np.linalg.norm(P[S[0]] - c)
            if r < best_r and np.all(np.linalg.norm(P - c, axis=1) <= r * (1 + tol) + tol):
                best_c, best_r = c, r
    return best_c, best_r


class MEBSMOTE:
    def __init__(self, k=5, random_state=None):
        self.k, self.random_state = k, random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        minority, _, m = _minority_split(y)
        if m == 0:
            return X, y
        Xm = X[y == minority].astype(float)
        nbrs = _neighbours_without_self(Xm, self.k)
        selected = rng.choice(len(Xm), size=m, replace=m > len(Xm))
        delta = rng.uniform(0, 1, size=m)
        centres = {i: minimum_enclosing_ball(Xm[nbrs[i]])[0] for i in np.unique(selected)}
        C = np.array([centres[i] for i in selected])
        X_new = Xm[selected] + delta[:, None] * (C - Xm[selected])
        return _append(X, y, X_new, minority)


# --------------------------------------------------------------------------------------------
# Self-paced Ensemble (Liu et al., ICDE 2020, arXiv:1909.03500), Algorithm 1, with the paper's
# experimental settings: absolute-error hardness |F(x) - y|, k = 20 equal-width hardness bins
# over [0, 1], and n = 10 base classifiers (SPE_10 in the paper's main tables).
#   f_0 is trained on all minority samples P plus a random majority subset of size |P|.
#   For i = 1 .. n-1: F_i = mean of f_0..f_{i-1}; majority samples are put into k bins by
#   hardness; bin weight p_l = 1 / (h_l + alpha) with h_l the mean hardness of bin l and
#   alpha = tan(i * pi / (2n)); bin l contributes |P| * p_l / sum(p) samples. Empty bins get
#   weight 0, and a bin with fewer samples than its share passes the shortfall on to the other
#   bins in proportion to their weights, so every member sees exactly |P| majority samples (the
#   paper does not cover these cases; see _allocate); f_i is
#   trained on P plus that subset. The final score is the mean of all n members.
# Member scores: predict_proba[:, 1] for logistic regression and XGBoost. LinearSVC has no
# predict_proba; its member score is sigmoid(decision_function), a monotone map of its decision
# value with sigmoid(0) = 0.5, so a member's 0.5 threshold equals the SVM's own decision rule.
# The paper does not say how probabilities were obtained for SVM.
# predict_proba returns the mean member score (used for AUC-ROC); predict uses threshold 0.5.
# --------------------------------------------------------------------------------------------
def _allocate(total, weights, capacity):
    """Split `total` samples over bins in proportion to `weights`, never exceeding a bin's
    `capacity`. A bin's shortfall is passed on to the other bins, again in proportion to their
    weights, so exactly `total` samples are drawn (the paper's |P| majority samples per member;
    it does not say what happens when a bin is too small)."""
    alloc = np.zeros(len(weights), dtype=int)
    active = (weights > 0) & (capacity > 0)
    while alloc.sum() < total and active.any():
        rem = total - alloc.sum()
        w = np.where(active, weights, 0.0)
        add = np.minimum(np.floor(rem * w / w.sum()).astype(int), capacity - alloc)
        if add.sum() == 0:                       # fewer than one sample per bin left
            for b in np.argsort(-w):
                if rem == 0 or not active[b]:
                    break
                if alloc[b] < capacity[b]:
                    alloc[b] += 1
                    rem -= 1
        else:
            alloc += add
        active &= alloc < capacity
    return alloc


def member_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


class SelfPacedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=10, k_bins=20, random_state=None):
        self.estimator, self.n_estimators = estimator, n_estimators
        self.k_bins, self.random_state = k_bins, random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.array([0, 1])
        minority, majority, _ = _minority_split(y)
        P = np.where(y == minority)[0]
        N = np.where(y == majority)[0]
        n_p = len(P)
        self.estimators_, self.subset_sizes_, self.subset_counts_ = [], [], []

        def train(maj_idx):
            idx = np.concatenate([P, maj_idx])
            self.estimators_.append(clone(self.estimator).fit(X[idx], y[idx]))
            self.subset_sizes_.append(len(idx))
            self.subset_counts_.append(np.bincount(y[idx].astype(int), minlength=2))

        train(rng.choice(N, size=min(n_p, len(N)), replace=False))
        score_sum = member_score(self.estimators_[0], X[N])
        for i in range(1, self.n_estimators):
            F = score_sum / i                       # P(class 1) from the current ensemble
            hardness = np.abs(F - majority)
            bins = np.minimum((hardness * self.k_bins).astype(int), self.k_bins - 1)
            alpha = np.tan(i * np.pi / (2 * self.n_estimators))
            weights = np.zeros(self.k_bins)
            for b in range(self.k_bins):
                in_bin = bins == b
                if in_bin.any():
                    weights[b] = 1.0 / (hardness[in_bin].mean() + alpha)
            quota = _allocate(min(n_p, len(N)), weights, np.bincount(bins, minlength=self.k_bins))
            chosen = []
            for b in range(self.k_bins):
                if quota[b] > 0:
                    chosen.append(rng.choice(N[bins == b], size=quota[b], replace=False))
            train(np.concatenate(chosen))
            score_sum += member_score(self.estimators_[-1], X[N])
        return self

    def predict_proba(self, X):
        s = np.mean([member_score(m, X) for m in self.estimators_], axis=0)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


OVERSAMPLERS = {
    "simplicial": lambda seed: SimplicialSMOTE(k=10, p=3, random_state=seed),
    "mgs": lambda seed: MGS(random_state=seed),
    # Cross-dataset runs: identical algorithm with the faster exact brute-force neighbour search.
    "mgs_brute": lambda seed: MGS(random_state=seed, knn_algorithm="brute"),
    "meb": lambda seed: MEBSMOTE(k=5, random_state=seed),
}


def resample(technique, X, y, seed):
    """Training data the classifier is fitted on. Oversamplers resample (training data only);
    baseline and SPE return the data unchanged (SPE undersamples inside fit_model)."""
    if technique in OVERSAMPLERS:
        return OVERSAMPLERS[technique](seed).fit_resample(X, y)
    if technique in ("baseline", "spe"):
        return X, y
    raise ValueError(f"unknown technique: {technique}")


def fit_model(technique, clf, X, y, seed):
    """Fit on (possibly resampled) training data. Returns (model, rows seen, class counts seen).

    For SPE, rows/counts are those of one member's training subset (the first member: all
    minority samples plus an equal-size random majority subset).
    """
    if technique == "spe":
        model = SelfPacedEnsemble(clf, n_estimators=10, k_bins=20, random_state=seed).fit(X, y)
        return model, model.subset_sizes_[0], model.subset_counts_[0]
    return clf.fit(X, y), len(y), np.bincount(y.astype(int), minlength=2)


def fit_with_technique(technique, clf, X, y, seed):
    """Resample, then fit. Returns (fitted model, number of training rows the classifier saw)."""
    X_res, y_res = resample(technique, X, y, seed)
    model, n_fit, _ = fit_model(technique, clf, X_res, y_res, seed)
    return model, n_fit
