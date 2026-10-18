"""Check for exact-duplicate texts shared by DiFrauD and UNIDECOR.
"""
import pandas as pd

from data_loading import ROOT, load_difraud, load_unidecor

OUT = ROOT / "results" / "data_stats"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    dif = load_difraud()
    uni = load_unidecor()
    dif["key"] = dif["text"].fillna("").str.strip()
    uni["key"] = uni["text"].fillna("").str.strip()

    shared = set(dif["key"]) & set(uni["key"])
    shared.discard("")

    summary = pd.DataFrame([{
        "shared_unique_texts": len(shared),
        "difraud_rows_affected": int(dif["key"].isin(shared).sum()),
        "unidecor_rows_affected": int(uni["key"].isin(shared).sum()),
        "difraud_rows_total": len(dif),
        "unidecor_rows_total": len(uni),
    }])
    summary.to_csv(OUT / "cross_duplicates_summary.csv", index=False)

    d = dif[dif["key"].isin(shared)][["key", "domain"]].drop_duplicates()
    u = uni[uni["key"].isin(shared)][["key", "domain"]].drop_duplicates()
    pairs = (d.merge(u, on="key", suffixes=("_difraud", "_unidecor"))
              .groupby(["domain_difraud", "domain_unidecor"]).size()
              .rename("shared_unique_texts").reset_index())
    pairs.to_csv(OUT / "cross_duplicates_by_pair.csv", index=False)

    print(summary.to_string(index=False))
    print(pairs.to_string(index=False) if len(pairs) else "No shared texts.")
