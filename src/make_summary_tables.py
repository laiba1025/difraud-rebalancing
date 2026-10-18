"""Final summary tables, built only from the saved result files (no model is run).
"""
import argparse

import pandas as pd

from data_loading import DIFRAUD_DOMAINS, ROOT
from metrics import METRICS

TECH_ORDER = ["baseline", "simplicial", "mgs", "meb", "spe"]
TECH_NAMES = {"baseline": "no rebalancing", "simplicial": "Simplicial SMOTE",
              "mgs": "MGS", "meb": "MEB-SMOTE", "spe": "Self-paced Ensemble"}
SETTING_ORDER = DIFRAUD_DOMAINS + ["pooled"]
STAT_COLS = [f"{m}_{s}" for m in METRICS for s in ["mean", "std"]]


def add_deltas(df, keys):
    """Change of every metric mean versus the baseline row with the same keys."""
    base = df[df["technique"] == "baseline"][keys + [f"{m}_mean" for m in METRICS]]
    out = df.merge(base, on=keys, how="left", suffixes=("", "_baseline"))
    for m in METRICS:
        out[f"delta_{m}"] = out[f"{m}_mean"] - out[f"{m}_mean_baseline"]
    return out.drop(columns=[f"{m}_mean_baseline" for m in METRICS])


def order(df, by):
    df = df.copy()
    if "technique" in df:
        df["technique_name"] = df["technique"].map(TECH_NAMES)
        df["technique"] = pd.Categorical(df["technique"], TECH_ORDER, ordered=True)
    if "setting" in df:
        df["setting"] = pd.Categorical(df["setting"], SETTING_ORDER, ordered=True)
    return df.sort_values(by).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-domain", default="results/in_domain")
    ap.add_argument("--rebalancing", default="results/in_domain_rebalancing")
    ap.add_argument("--cross", default="results/cross_dataset")
    ap.add_argument("--out", default="results/summary_tables")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(ROOT / args.in_domain / "summary.csv")
    reb = pd.read_csv(ROOT / args.rebalancing / "summary.csv")
    cross = pd.read_csv(ROOT / args.cross / "summary.csv")
    notes = reb.groupby("setting")["setting_note"].first().fillna("") if "setting_note" in reb else {}

    # In-domain baseline.
    a = base[base["technique"] == "baseline"].copy()
    a["setting_note"] = a["setting"].map(notes).fillna("")
    a = order(a, ["setting", "classifier"])
    a[["setting", "classifier", "train_minority_class", "n_seeds"] + STAT_COLS + ["setting_note"]].to_csv(
        out / "in_domain_baseline.csv", index=False, float_format="%.4f")

    # In-domain rebalancing, with baseline rows and changes versus baseline.
    b = pd.concat([base[base["technique"] == "baseline"], reb[reb["technique"] != "baseline"]],
                  ignore_index=True)
    b["setting_note"] = b["setting"].map(notes).fillna("")
    b = order(add_deltas(b, ["setting", "classifier"]), ["setting", "classifier", "technique"])
    b[["setting", "classifier", "technique", "technique_name", "train_minority_class", "n_seeds"]
      + STAT_COLS + [f"delta_{m}" for m in METRICS] + ["setting_note"]].to_csv(
        out / "in_domain_rebalancing.csv", index=False, float_format="%.4f")

    # Cross-dataset (pooled source models), incl. the labelled UNIDECOR in-domain reference.
    keys = ["direction", "mode", "source_model", "classifier", "target_group"]
    c = add_deltas(cross, keys)
    c["result_type"] = c["mode"].map(lambda m: "in-domain reference (not cross-dataset)"
                                     if m == "in_domain_reference" else "cross-dataset")
    c["target_group_order"] = (c["target_group"] != "ALL").astype(int)
    cols = (["result_type", "direction", "mode", "source_model", "target_group", "classifier",
             "technique", "technique_name", "n", "n_deceptive", "pct_deceptive", "low_n", "n_seeds"]
            + STAT_COLS + [f"delta_{m}" for m in METRICS])
    pooled = order(c[c["source_model"] == "pooled"],
                   ["result_type", "direction", "mode", "target_group_order", "target_group",
                    "classifier", "technique"])
    pooled[cols].to_csv(out / "cross_dataset.csv", index=False, float_format="%.4f")

    # Frozen per-domain DiFrauD models -> UNIDECOR.
    per = c[(c["direction"] == "DiFrauD->UNIDECOR") & (c["mode"] == "frozen")
            & (c["source_model"] != "pooled")].copy()
    per["source_order"] = per["source_model"].map({d: i for i, d in enumerate(DIFRAUD_DOMAINS)})
    per = order(per, ["source_order", "target_group_order", "target_group", "classifier", "technique"])
    per[cols].to_csv(out / "cross_dataset_frozen_per_domain.csv", index=False, float_format="%.4f")

    for f in sorted(out.glob("*.csv")):
        print(f"{f.name}: {len(pd.read_csv(f))} rows")


if __name__ == "__main__":
    main()
