"""Inspect DiFrauD and UNIDECOR (sample counts, class ratio, text length, splits).
"""
import json
import sys
from importlib.metadata import version

import pandas as pd

from data_loading import ROOT, difraud_revision, load_difraud, load_unidecor_raw

OUT = ROOT / "results" / "data_stats"
OUT.mkdir(parents=True, exist_ok=True)

# All 14 components listed in the UNIDECOR repo (get_consolidated_data.py).
UNIDECOR_ALL = ["DEC_OP", "BLT_C", "OP_SPAM", "TRIAL_DE", "OPEN_DOMAIN", "CROSS_CULT_DE",
                "DE_REV_2014", "DE_REV_2018", "ONLINE_DE", "DIPLOMACY_CONVO", "MU3D",
                "MAFIASCUM", "BLUFF", "BOX_OF_LIES"]


def stats(df, is_deceptive, extra=None):
    text = df["text"].fillna("").astype(str)
    n = len(df)
    n_dec = int(is_deceptive.sum())
    row = {
        "n": n,
        "n_truthful": n - n_dec,
        "n_deceptive": n_dec,
        "pct_deceptive": round(100 * n_dec / n, 2) if n else None,
        "avg_chars": round(text.str.len().mean(), 1),
        "avg_words": round(text.str.split().str.len().mean(), 1),
        "n_empty_text": int((text.str.strip() == "").sum()),
    }
    row.update(extra or {})
    return row


def inspect_difraud():
    rev = difraud_revision()
    df = load_difraud(rev)
    rows = []
    for (dom, split), g in df.groupby(["domain", "split"], sort=False):
        rows.append({"domain": dom, "split": split, **stats(g, g["label"] == 1)})
    for dom, g in df.groupby("domain", sort=False):
        rows.append({"domain": dom, "split": "all", **stats(g, g["label"] == 1)})
    for split, g in df.groupby("split", sort=False):
        rows.append({"domain": "POOLED", "split": split, **stats(g, g["label"] == 1)})
    rows.append({"domain": "POOLED", "split": "all", **stats(df, df["label"] == 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "difraud_stats.csv", index=False)
    return out, rev


def inspect_unidecor():
    df = load_unidecor_raw()
    dec = df["truth_label"] == "D"
    rows = []
    for src, g in df.groupby("source", sort=False):
        splits = sorted(g["split"].dropna().unique().tolist())
        folds = sorted(g["fold"].dropna().astype(str).unique().tolist())
        rows.append({"source": src, **stats(g, dec[g.index], {
            "domain": g["domain"].iloc[0], "mode": g["mode"].iloc[0],
            "languages": ",".join(sorted(g["language"].unique())),
            "splits": ",".join(splits) or "none",
            "folds": ",".join(folds) or "none",
        })})
    rows.append({"source": "POOLED", **stats(df, dec)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "unidecor_stats.csv", index=False)

    present = set(df["source"])
    pd.DataFrame({"source": UNIDECOR_ALL,
                  "obtained": [s in present for s in UNIDECOR_ALL]}
                 ).to_csv(OUT / "unidecor_components.csv", index=False)
    (df.groupby(["source", "language"]).size().rename("n").reset_index()
       .to_csv(OUT / "unidecor_languages.csv", index=False))
    return out, present


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    d, rev = inspect_difraud()
    u, present = inspect_unidecor()
    print("DiFrauD revision:", rev)
    print(d.to_string(index=False))
    print()
    print(u.to_string(index=False))
    print("\nUNIDECOR missing components:", [s for s in UNIDECOR_ALL if s not in present])
    meta = {"difraud_repo": "redasers/difraud", "difraud_revision": rev,
            "python": sys.version.split()[0],
            **{p: version(p) for p in ["pandas", "huggingface_hub", "convokit"]}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
