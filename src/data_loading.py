"""Loaders for DiFrauD and UNIDECOR.

"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

DIFRAUD_REPO = "redasers/difraud"
DIFRAUD_DOMAINS = ["fake_news", "job_scams", "phishing", "political_statements",
                   "product_reviews", "sms", "twitter_rumours"]
DIFRAUD_SPLITS = ["train", "validation", "test"]
# Pinned commit of the HF dataset repo used for all experiments.
DIFRAUD_REVISION = "f8fa04aff083aa22c6878d005b4655e34a4cbd85"

UNIDECOR_JSON = ROOT / "unidecor_repo" / "datasets" / "consolidated_deception_data.json"


def difraud_revision():
    return DIFRAUD_REVISION or HfApi().dataset_info(DIFRAUD_REPO).sha


def load_difraud(revision=None):
    """Return one DataFrame with columns text, label (1=deceptive), domain, split."""
    revision = revision or difraud_revision()
    frames = []
    for dom in DIFRAUD_DOMAINS:
        for split in DIFRAUD_SPLITS:
            path = hf_hub_download(DIFRAUD_REPO, f"{dom}/{split}.jsonl",
                                   repo_type="dataset", revision=revision)
            df = pd.read_json(path, lines=True, dtype={"text": str})
            df["label"] = df["label"].astype(int)
            df["domain"], df["split"] = dom, split
            frames.append(df[["text", "label", "domain", "split"]])
    return pd.concat(frames, ignore_index=True)


def load_unidecor_raw(path=UNIDECOR_JSON):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def load_unidecor(path=UNIDECOR_JSON):
    """UNIDECOR mapped to DiFrauD's format: text, label (D=1, T=0), domain (= source).

    Non-English texts are dropped (only CROSS_CULT_DE has any: ES_mx and RO), since
    DiFrauD is English-only. Splits are created later (see README).
    """
    df = load_unidecor_raw(path)
    df = df[df["language"] == "EN"]
    out = pd.DataFrame({"text": df["text"].astype(str),
                        "label": df["truth_label"].map({"D": 1, "T": 0}).astype(int),
                        "domain": df["source"],
                        "text_ID": df["text_ID"].astype(str),
                        "official_split": df["split"]})
    return out.reset_index(drop=True)


def unidecor_splits(u, seed=42):
    """Train/validation/test assignment for UNIDECOR (approved design).

    DIPLOMACY_CONVO keeps its official split (validation -> "validation"). Every other component
    gets a stratified 80/10/10 split with the given seed. Returns a Series aligned with u.
    """
    from sklearn.model_selection import train_test_split
    split = pd.Series(index=u.index, dtype=object)
    for comp, g in u.groupby("domain"):
        if comp == "DIPLOMACY_CONVO":
            split[g.index] = g["official_split"].str.lower()
            continue
        tr, rest = train_test_split(g.index, test_size=0.2, stratify=g["label"], random_state=seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=g.loc[rest, "label"], random_state=seed)
        split[tr], split[va], split[te] = "train", "validation", "test"
    return split
