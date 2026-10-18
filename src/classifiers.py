"""The three classifiers, with library-default hyperparameters.
Only random_state is set; XGBoost additionally uses
tree_method="hist", as approved, and all CPU threads.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

CLASSIFIERS = ["logreg", "linear_svm", "xgboost"]


def make_classifier(name, seed):
    if name == "logreg":
        return LogisticRegression(random_state=seed)
    if name == "linear_svm":
        return LinearSVC(random_state=seed)
    if name == "xgboost":
        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def scores(model, X):
    """Continuous score for the deceptive class (1), used for AUC-ROC.

    LinearSVC has no predict_proba, so its decision_function is used.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
