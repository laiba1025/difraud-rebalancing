"""Text features: TF-IDF (word 1-2-grams) followed by TruncatedSVD to 300 dimensions.
"""
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

FEATURE_PARAMS = {
    "tfidf": {"ngram_range": (1, 2), "sublinear_tf": True, "min_df": 2, "max_features": 50000},
    "svd": {"n_components": 300, "random_state": 42},
}


def fit_features(train_texts):
    """Fit TF-IDF + SVD on training texts. Returns (fitted pipeline, train features)."""
    pipe = make_pipeline(TfidfVectorizer(**FEATURE_PARAMS["tfidf"]),
                         TruncatedSVD(**FEATURE_PARAMS["svd"]))
    Z = pipe.fit_transform(train_texts)
    # stop_words_ holds every pruned n-gram; it is only for introspection and can be very
    # large, so it is dropped before the pipeline is saved (documented as safe in sklearn).
    pipe[0].stop_words_ = None
    return pipe, Z.astype("float32")
