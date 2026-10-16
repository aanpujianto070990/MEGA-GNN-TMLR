r"""Per-transaction metadata for the thesis analysis, one row per raw transaction.

Columns:
  edge_id        row index in the raw *_Trans.csv (same as `edge_id` in the
                 processed data and in scores_*.npz)
  timestamp      raw timestamp string
  is_laundering  label
  split          train / val / test, from the same daily split as AMLDataset
  period         active (date <= --active-end) or tail
  pattern        laundering pattern type from *_Patterns.txt, empty if none
  attempt        index of the laundering attempt in *_Patterns.txt, -1 if none
  delta_t        seconds since the sender's previous incoming transaction
                 (strictly earlier), -1 if none; same as AMLDataset's delta_t

Usage (from the repository root):
  python thesis/metadata.py --trans HI-Small_Trans.csv \
      --patterns HI-Small_Patterns.txt --out meta_Small-HI.csv.gz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MegaGNN.datasets.aml_dataset import causal_delta_t, daily_split, format_transactions  # noqa: E402

# Last day of the active period: after it, normal transactions almost stop
# while laundering transactions continue (proposal, Bab I and Bab II).
ACTIVE_END = "2022-09-10"


def read_patterns(path):
    """One row per transaction line in *_Patterns.txt: (line, pattern, attempt)."""
    rows, pattern, attempt = [], None, -1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                # "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out" -> "fan-out"
                pattern = line.split(" - ", 1)[1].split(":")[0].strip().lower()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                pattern = None
            else:
                rows.append((line, pattern, attempt))
    return pd.DataFrame(rows, columns=["line", "pattern", "attempt"])


def build_metadata(trans_path, patterns_path, active_end=ACTIVE_END):
    raw = pd.read_csv(trans_path, dtype=str)
    fmt = format_transactions(raw)

    # Split and delta_t exactly as AMLDataset.process computes them.
    ts = fmt["Timestamp"].to_numpy() - fmt["Timestamp"].min()
    train_inds, val_inds, test_inds, days = daily_split(torch.tensor(ts, dtype=torch.float32))
    split = np.empty(len(fmt), dtype=object)
    for name, inds in [("train", train_inds), ("val", val_inds), ("test", test_inds)]:
        split[inds.numpy()] = name
    delta_t = causal_delta_t(fmt["from_id"].to_numpy(), fmt["to_id"].to_numpy(), ts)

    meta = pd.DataFrame({
        "edge_id": fmt["EdgeID"].to_numpy(),
        "split": split,
        "delta_t": delta_t,
    }).sort_values("edge_id").reset_index(drop=True)

    meta.insert(1, "timestamp", raw["Timestamp"].to_numpy())
    meta.insert(2, "is_laundering", raw["Is Laundering"].astype(int).to_numpy())
    dates = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M").dt.normalize()
    meta.insert(4, "period", np.where(dates <= pd.Timestamp(active_end), "active", "tail"))

    # Pattern transactions are raw CSV lines; all of them are labelled
    # laundering, so only those rows need a lookup key.
    patterns = read_patterns(patterns_path)
    illicit = raw.index[raw["Is Laundering"] == "1"]
    keys = pd.Series(raw.loc[illicit].agg(",".join, axis=1).to_numpy(), index=illicit)
    matched = patterns.merge(keys.rename("line").reset_index(), on="line", how="left")
    n_missing = int(matched["index"].isna().sum())
    n_multi = int(matched.duplicated("line", keep=False).sum() - patterns.duplicated("line", keep=False).sum())
    if n_missing or n_multi:
        raise ValueError(f"{n_missing} pattern lines without a matching transaction, "
                         f"{n_multi} matching more than one transaction")

    meta["pattern"] = ""
    meta["attempt"] = -1
    rows = matched["index"].astype(int).to_numpy()  # raw row == edge_id == meta row
    meta.loc[rows, "pattern"] = matched["pattern"].to_numpy()
    meta.loc[rows, "attempt"] = matched["attempt"].to_numpy()
    return meta, days


def describe(meta, days):
    print(f"days per split (train / val / test): {[len(d) for d in days]} -> {days}")
    tab = meta.groupby("split").agg(transactions=("edge_id", "size"), illicit=("is_laundering", "sum"))
    tab["share_%"] = 100 * tab["transactions"] / len(meta)
    print(tab.loc[["train", "val", "test"]].round(2).to_string())

    test = meta[meta["split"] == "test"]
    by_period = test.groupby("period").agg(transactions=("edge_id", "size"), illicit=("is_laundering", "sum"))
    by_period["illicit_share_of_test_%"] = 100 * by_period["illicit"] / test["is_laundering"].sum()
    print("\ntest split by period:")
    print(by_period.round(2).to_string())

    ill = meta[meta["is_laundering"] == 1]
    print(f"\nillicit transactions in a pattern attempt: {(ill['pattern'] != '').sum()} of {len(ill)} "
          f"({100 * (ill['pattern'] != '').mean():.1f}%)")
    pat = meta[meta["pattern"] != ""]
    print(pat.groupby(["pattern", "split"])["attempt"].nunique().unstack(fill_value=0).to_string())

    has = meta["delta_t"] >= 0
    print(f"\ndelta_t: sender has an earlier incoming transaction for {100 * has.mean():.2f}% of transactions; "
          f"median {meta.loc[has, 'delta_t'].median() / 3600:.2f} h "
          f"(illicit {meta.loc[has & (meta['is_laundering'] == 1), 'delta_t'].median() / 3600:.2f} h, "
          f"licit {meta.loc[has & (meta['is_laundering'] == 0), 'delta_t'].median() / 3600:.2f} h)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trans", required=True, help="raw *_Trans.csv")
    parser.add_argument("--patterns", required=True, help="matching *_Patterns.txt")
    parser.add_argument("--out", required=True, help="output .csv or .csv.gz")
    parser.add_argument("--active-end", default=ACTIVE_END, help="last day of the active period (YYYY-MM-DD)")
    args = parser.parse_args()

    meta, days = build_metadata(args.trans, args.patterns, args.active_end)
    describe(meta, days)
    meta.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(meta)} rows)")


if __name__ == "__main__":
    main()
