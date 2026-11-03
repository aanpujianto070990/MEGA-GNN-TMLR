r"""Stage-6 analysis of the thesis main experiment (Multi-GIN+EU on AMLworld HI/LI-Small).

Reads the per-run outputs of Eksperimen_Utama_Kaggle.ipynb (run_summary.json,
scores_test.npz, optionally scores_val.npz, val/stats.json) and the
per-transaction metadata written by thesis/metadata.py, and writes to --out:

  runs.csv        one row per run: validity, metrics per test period, training cost
  summary.csv     mean / SD / n per dataset x configuration (x run variant)
  tests.csv       paired Wilcoxon signed-rank tests, Holm-corrected per family and metric
  patterns.csv    per run and laundering pattern: recall at threshold 0.5 and in the
                  0.5% alarm budget, share of attempts with at least one alarm
  cost.csv        compute cost per configuration and overhead vs the baseline
  pr_<dataset>.png  mean precision-recall curve per configuration (test, active period)
  laporan.md      the main tables in Markdown (Indonesian)

Metrics are in percent. Periods: all = whole test split, active = up to
2022-09-10, tail = after it (the `period` column of the metadata). Recall@k:
share of the illicit transactions of the period among the k = 0.1% / 0.5% of
that period's transactions with the highest scores. A run is valid when it is a
baseline, or a temporal run with the delta_t horizon (`temporal.horizon_days`
> 0 in run_summary.json); older temporal runs are kept as variant
`no_horizon` / `old_init` for the preliminary analysis only.

Usage (Kaggle; CPU is enough):
  python thesis/analysis.py --results /kaggle/input/<bundle>/runs \
      --meta-dir /kaggle/input/<bundle>/meta --out /kaggle/working/analisis [--val-threshold]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, precision_recall_curve

BACKBONE = "Multi-GIN"
DATASETS = ["Small-LI", "Small-HI"]
CONFIGS = ["base", "t2v", "sin", "decay", "full"]
PERIODS = ["all", "active", "tail"]
BUDGETS = {"0.1%": 0.001, "0.5%": 0.005}
RUN_DIR_RE = re.compile(r"AML-(Small-(?:HI|LI))-(Multi-GIN|MEGA-GIN)-(base|full|t2v|sin|decay)-gpu\d+$")

# Paired comparisons (family, configuration, reference). H1 = full vs baseline
# (seeds 1-10); the rest are the ablation questions (seeds 1-5). The confirmatory
# test of hypothesis H1 (proposal, Bab II) is one pre-specified test, full vs
# baseline on F1 of the active test period of LI-Small: it is not corrected.
# Every other test is corrected with Holm's method per family and metric.
CONFIRMATORY = ("H1", "Small-LI", "full vs base", "f1_active")
COMPARISONS = [
    ("H1", "full", "base"),
    ("ablasi", "t2v", "base"),
    ("ablasi", "sin", "base"),
    ("ablasi", "decay", "base"),
    ("ablasi", "t2v", "sin"),
    ("ablasi", "full", "t2v"),
    ("ablasi", "full", "decay"),
]
TEST_METRICS = ["f1_all", "f1_active", "ap_all", "ap_active", "recall_0.1%_active", "recall_0.5%_active"]
THRESHOLD_METRICS = ["f1valthr_all", "f1valthr_active"]
SUMMARY_METRICS = ["f1_all", "f1_active", "f1_tail", "precision_all", "recall_all", "ap_all", "ap_active",
                   "recall_0.1%_all", "recall_0.5%_all", "recall_0.1%_active", "recall_0.5%_active"]


# ----------------------------------------------------------------------------- runs

def run_variant(cfg, temporal):
    """valid | no_horizon | old_init (see the module docstring)."""
    if cfg == "base" or temporal.get("horizon_days", 0) > 0:
        return "valid"
    if temporal.get("encoding", "none") != "none" and temporal.get("proj_init") != "zero":
        return "old_init"
    return "no_horizon"


def find_runs(roots):
    """{(dataset, config, seed, variant): (run dir, run_summary dict)}; the first copy of a run wins."""
    runs = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "run_summary.json"), recursive=True)):
            seed_dir = os.path.dirname(path)
            m = RUN_DIR_RE.search(os.path.basename(os.path.dirname(seed_dir)))
            if not m or m.group(2) != BACKBONE:
                continue
            ds, _, cfg = m.groups()
            with open(path) as f:
                summary = json.load(f)
            key = (ds, cfg, int(os.path.basename(seed_dir)), run_variant(cfg, summary.get("temporal", {})))
            if key in runs:
                old = runs[key][1]["test"]["f1"]
                if abs(old - summary["test"]["f1"]) > 1e-9:
                    print(f"WARNING: {key} found twice with different test F1 ({old} vs {summary['test']['f1']}); "
                          f"keeping {runs[key][0]}")
                continue
            runs[key] = (seed_dir, summary)
    return runs


# ----------------------------------------------------------------------------- metadata

class Meta:
    """Per-transaction metadata of one dataset, indexed by edge_id (= raw CSV row)."""

    def __init__(self, path):
        df = pd.read_csv(path, usecols=["edge_id", "is_laundering", "split", "period", "pattern", "attempt"],
                         dtype={"split": "category", "period": "category", "pattern": "category"},
                         keep_default_na=False)
        if not np.array_equal(df["edge_id"].to_numpy(), np.arange(len(df))):
            raise ValueError(f"{path}: edge_id is not 0..N-1 in order")
        self.y = df["is_laundering"].to_numpy(np.int8)
        self.split = df["split"].to_numpy(dtype=object)
        self.n_test = int((self.split == "test").sum())
        self.tail = (df["period"] == "tail").to_numpy()
        self.pattern = df["pattern"].cat.codes.to_numpy()
        self.pattern_names = ["(tanpa pola)" if p == "" else p for p in df["pattern"].cat.categories]
        self.attempt = df["attempt"].to_numpy(np.int64)


def load_scores(path, meta, split):
    """(edge_id, y, score) of a scores_<split>.npz, one entry per transaction, checked against the metadata."""
    with np.load(path) as z:
        eid = z["edge_id"].astype(np.int64)
        y = z["y"].astype(np.int8)
        s = z["score"].astype(np.float64)
    if s.ndim == 2:  # log-probabilities of two classes
        s = np.exp(s[:, 1])
    eid, first = np.unique(eid, return_index=True)
    y, s = y[first], s[first]
    if (meta.y[eid] != y).any():
        raise ValueError(f"{path}: labels differ from the metadata")
    if not (meta.split[eid] == split).all():
        raise ValueError(f"{path}: transactions outside the {split} split")
    return eid, y, s


# ----------------------------------------------------------------------------- metrics

def metrics(y, s, pred, budgets=True):
    """Metrics in percent for labels y, scores s and alarms pred (bool)."""
    n, pos = len(y), int(y.sum())
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / pos if pos else np.nan
    f1 = 2 * prec * rec / (prec + rec) if pos and prec + rec > 0 else 0.0
    out = {"n": n, "illicit": pos, "f1": 100 * f1, "precision": 100 * prec, "recall": 100 * rec,
           "ap": 100 * average_precision_score(y, s) if 0 < pos < n else np.nan}
    # Ties in the score keep the edge_id order (arrays are sorted by edge_id).
    order = np.argsort(-s, kind="stable")
    for tag, frac in BUDGETS.items():
        k = max(1, int(np.floor(frac * n + 0.5)))
        out[f"recall_{tag}"] = 100 * y[order[:k]].sum() / pos if budgets and pos else np.nan
    return out


def best_threshold(y, s):
    """Score threshold (alarm = score >= threshold) that maximises F1 on (y, s)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]  # cut only between different scores
    tp, fp, ss = tp[last], fp[last], ss[last]
    prec, rec = tp / (tp + fp), tp / max(int(ys.sum()), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(ss[i]), float(100 * f1[i])


def interp_pr(y, s, grid):
    """Interpolated precision (max precision at recall >= r) at each recall r of grid."""
    p, r, _ = precision_recall_curve(y, s)
    order = np.argsort(r, kind="stable")
    r, p = r[order], p[order]
    pmax = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.searchsorted(r, grid, side="left")
    return np.where(idx < len(r), pmax[np.minimum(idx, len(r) - 1)], 0.0)


def first_positive_epoch(seed_dir):
    """First epoch whose validation F1 is > 0 (from val/stats.json), or None."""
    path = os.path.join(seed_dir, "val", "stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("f1", 0) > 0:
                    return int(rec["epoch"])
    return None


def evaluate_run(key, seed_dir, summary, meta, val_threshold, grid):
    """(runs.csv row, patterns.csv rows, PR curve of the active period or None)."""
    ds, cfg, seed, variant = key
    t = summary.get("temporal", {})
    train_s = float(np.sum(summary.get("train_epoch_s", [])))
    val_s = float(np.sum(summary.get("val_eval_s", [])))
    row = {"dataset": ds, "config": cfg, "seed": seed, "variant": variant, "valid": variant == "valid",
           "run_dir": seed_dir, "epochs_run": summary.get("epochs_run"), "best_epoch": summary.get("best_epoch"),
           "stopped_early": summary.get("stopped_early"), "max_epoch": summary.get("max_epoch"),
           "first_val_f1_pos_epoch": first_positive_epoch(seed_dir),
           "val_f1": 100 * summary["val"]["f1"], "summary_test_f1": 100 * summary["test"]["f1"],
           "summary_test_ap": 100 * summary["test"].get("pr-auc", np.nan),
           "params": summary.get("params"), "horizon_days": t.get("horizon_days", 0.0),
           "lambda_per_day": t.get("lambda_per_day", np.nan),
           "epoch_train_s": float(np.mean(summary["train_epoch_s"])) if summary.get("train_epoch_s") else np.nan,
           "run_hours": (train_s + val_s + float(summary.get("test_eval_s") or 0)) / 3600,
           "peak_gpu_mem_mb": summary.get("peak_gpu_mem_allocated_mb"),
           "test_coverage": np.nan}
    path = os.path.join(seed_dir, "scores_test.npz")
    if not os.path.exists(path):
        print(f"WARNING: {key}: no scores_test.npz, only run_summary values")
        return row, [], None

    eid, y, s = load_scores(path, meta, "test")
    row["test_coverage"] = 100 * len(eid) / meta.n_test
    tail = meta.tail[eid]
    masks = {"all": np.ones(len(eid), bool), "active": ~tail, "tail": tail}
    pred = s > 0.5
    for period, m in masks.items():
        for name, v in metrics(y[m], s[m], pred[m], budgets=period != "tail").items():
            if name in ("n", "illicit") and period != "all":
                continue
            row[f"{name}_{period}"] = v
    if abs(row["f1_all"] - row["summary_test_f1"]) > 0.05:
        print(f"WARNING: {key}: F1 from scores ({row['f1_all']:.2f}) differs from run_summary "
              f"({row['summary_test_f1']:.2f})")

    if val_threshold:
        vpath = os.path.join(seed_dir, "scores_val.npz")
        if os.path.exists(vpath):
            _, vy, vs = load_scores(vpath, meta, "val")
            thr, vf1 = best_threshold(vy, vs)
            row["val_threshold"], row["val_f1_at_threshold"] = thr, vf1
            tpred = s >= thr
            for period in ("all", "active"):
                m = masks[period]
                row[f"f1valthr_{period}"] = metrics(y[m], s[m], tpred[m], budgets=False)["f1"]
        else:
            print(f"WARNING: {key}: no scores_val.npz, no validation threshold")

    # Laundering patterns (whole test split, illicit transactions only).
    order = np.argsort(-s, kind="stable")
    in_budget = np.zeros(len(s), bool)
    in_budget[order[:max(1, int(np.floor(BUDGETS["0.5%"] * len(s) + 0.5)))]] = True
    ill = y == 1
    df = pd.DataFrame({"pattern": np.asarray(meta.pattern_names, dtype=object)[meta.pattern[eid][ill]],
                       "attempt": meta.attempt[eid][ill], "alarm": pred[ill], "budget": in_budget[ill]})
    pats = []
    for pattern, g in df.groupby("pattern", sort=True):
        att = g[g["attempt"] >= 0].groupby("attempt")["alarm"].any()
        pats.append({"dataset": ds, "config": cfg, "seed": seed, "variant": variant, "pattern": pattern,
                     "n_illicit": len(g), "recall_thr": 100 * g["alarm"].mean(),
                     "recall_budget_0.5%": 100 * g["budget"].mean(),
                     "n_attempts": len(att), "attempts_alarmed": 100 * att.mean() if len(att) else np.nan})

    curve = interp_pr(y[masks["active"]], s[masks["active"]], grid)
    return row, pats, curve


# ----------------------------------------------------------------------------- statistics

def paired_test(d):
    """Wilcoxon signed-rank test of paired differences d (zeros dropped), two-sided."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    out = {"n_pairs": len(d), "mean_diff": d.mean() if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan,
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
           # smallest p an exact two-sided test can give with this many non-zero pairs
           "p_min_possible": min(1.0, 2.0 / 2 ** len(nz)) if len(nz) else np.nan,
           "W_plus": np.nan, "r_rank_biserial": np.nan, "p": np.nan}
    if len(nz):
        ranks = rankdata(np.abs(nz))
        wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        out["W_plus"], out["r_rank_biserial"] = wp, (wp - wm) / (wp + wm)
        try:
            out["p"] = float(wilcoxon(nz).pvalue)
        except ValueError:
            pass
    return out


def holm(p):
    """Holm-adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(np.isfinite(p))
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        adj[i] = running
    return adj


def run_tests(runs, metric_names):
    valid = runs[runs["valid"]]
    rows = []
    for family, cfg, ref in COMPARISONS:
        for ds in DATASETS:
            a = valid[(valid["dataset"] == ds) & (valid["config"] == cfg)].set_index("seed")
            b = valid[(valid["dataset"] == ds) & (valid["config"] == ref)].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            for metric in metric_names:
                if metric not in valid or not seeds:
                    continue
                d = (a.loc[seeds, metric] - b.loc[seeds, metric]).to_numpy(float)
                rows.append({"family": family, "dataset": ds, "comparison": f"{cfg} vs {ref}", "metric": metric,
                             "seeds": ",".join(map(str, seeds)), "mean_cfg": a.loc[seeds, metric].mean(),
                             "mean_ref": b.loc[seeds, metric].mean(), **paired_test(d)})
    tests = pd.DataFrame(rows)
    if len(tests):
        key = tests[["family", "dataset", "comparison", "metric"]].apply(tuple, axis=1)
        tests.insert(0, "confirmatory", key.map(lambda k: k == CONFIRMATORY).astype(bool))
        tests["p_holm"] = np.nan
        rest = tests[~tests["confirmatory"]]
        for _, idx in rest.groupby(["family", "metric"]).groups.items():
            tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p"].to_numpy())
        tests.loc[tests["confirmatory"], "p_holm"] = tests.loc[tests["confirmatory"], "p"]
    return tests


# ----------------------------------------------------------------------------- report

def md_table(df, fmt="{:.2f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_sd(group, col):
    v = group[col].dropna()
    if not len(v):
        return ""
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def plot_pr(curves, grid, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, no PR plots")
        return
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        drawn = False
        for cfg in CONFIGS:
            cs = [c for (d, k, _), c in curves.items() if d == ds and k == cfg]
            if cs:
                ax.plot(grid, 100 * np.mean(cs, axis=0), label=f"{cfg} (n={len(cs)})")
                drawn = True
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xlabel("Recall (periode aktif)")
        ax.set_ylabel("Presisi terinterpolasi (%)")
        ax.set_title(f"Kurva PR rerata, {ds}, test periode aktif")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pr_{ds}.png"), dpi=150)
        plt.close(fig)


def write_report(runs, summary, tests, patterns, cost, out_dir):
    valid = runs[runs["valid"]]
    lines = ["# Laporan analisis tahap 6", "",
             "Dihasilkan oleh `thesis/analysis.py`. Metrik dalam persen; periode aktif = data uji sampai "
             "10 Sep 2022. Semua uji berpasangan per seed (Wilcoxon dua sisi, koreksi Holm per keluarga dan metrik).",
             "", "## 1. Run yang dipakai", ""]
    counts = runs.groupby(["dataset", "config", "variant"]).size().reset_index(name="run")
    lines += [md_table(counts), ""]

    lines += ["## 2. Ringkasan metrik (run valid, rerata ± SD)", ""]
    for ds in DATASETS:
        rows = []
        for cfg in CONFIGS:
            g = valid[(valid["dataset"] == ds) & (valid["config"] == cfg)]
            if not len(g):
                continue
            r = {"konfigurasi": cfg, "n": len(g)}
            for col, label in [("f1_all", "F1"), ("f1_active", "F1 aktif"), ("precision_all", "Presisi"),
                               ("ap_all", "AP"), ("ap_active", "AP aktif"),
                               ("recall_0.5%_active", "Recall@0,5% aktif"), ("f1valthr_all", "F1 ambang val")]:
                if col in g:
                    r[label] = mean_sd(g, col)
            rows.append(r)
        if rows:
            lines += [f"### {ds}", "", md_table(pd.DataFrame(rows)), ""]

    lines += ["## 3. Uji Wilcoxon berpasangan + Holm", "",
              "`p_min_possible` = p terkecil yang mungkin dengan jumlah pasangan itu (5 pasangan: 0,0625; "
              "10 pasangan: 0,002).", ""]
    if len(tests):
        lines += ["Baris `confirmatory = True` adalah uji konfirmatori H1 (F1 periode aktif LI-Small, satu uji, "
                  "tanpa koreksi). Semua uji lain dikoreksi Holm per keluarga dan metrik.", ""]
        cols = ["confirmatory", "family", "dataset", "comparison", "metric", "n_pairs", "mean_cfg", "mean_ref",
                "mean_diff", "wins", "losses", "r_rank_biserial", "p", "p_holm", "p_min_possible"]
        lines += [md_table(tests[cols], fmt="{:.3f}"), ""]

    lines += ["## 4. Per pola pencucian uang (recall pada ambang 0,5, rerata seed)", ""]
    if len(patterns):
        vp = patterns[patterns["variant"] == "valid"]
        for ds in DATASETS:
            g = vp[vp["dataset"] == ds]
            if not len(g):
                continue
            piv = g.pivot_table(index="pattern", columns="config", values="recall_thr", aggfunc="mean")
            piv = piv[[c for c in CONFIGS if c in piv.columns]]
            n = g.groupby("pattern")["n_illicit"].first()
            piv.insert(0, "n transaksi ilegal (test)", n)
            lines += [f"### {ds}", "", md_table(piv.reset_index()), ""]

    lines += ["## 5. Biaya komputasi (rerata per run; dua run paralel di T4 x2)", "", md_table(cost), ""]

    pre = summary[summary["variant"] != "valid"]
    if len(pre):
        cols = [c for c in ["dataset", "config", "variant", "n", "f1_all_mean", "ap_all_mean", "f1_active_mean",
                            "ap_active_mean", "summary_test_f1_mean", "summary_test_ap_mean"] if c in pre]
        lines += ["## 6. Analisis pendahuluan: run temporal tanpa horizon (sudah diganti)", "",
                  md_table(pre[cols]), ""]
    with open(os.path.join(out_dir, "laporan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="folder(s) searched recursively for runs")
    parser.add_argument("--meta-dir", required=True, help="folder with meta_Small-HI.csv.gz and meta_Small-LI.csv.gz")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--val-threshold", action="store_true",
                        help="also report F1 at the threshold that maximises validation F1 (needs scores_val.npz)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    found = find_runs(args.results)
    if not found:
        sys.exit("no runs found under " + " ".join(args.results))
    print(f"{len(found)} runs found")
    grid = np.linspace(0, 1, 201)
    rows, pats, curves = [], [], {}
    for ds in DATASETS:
        keys = sorted(k for k in found if k[0] == ds)
        if not keys:
            continue
        meta = Meta(os.path.join(args.meta_dir, f"meta_{ds}.csv.gz"))
        for key in keys:
            seed_dir, summary = found[key]
            row, p, curve = evaluate_run(key, seed_dir, summary, meta, args.val_threshold, grid)
            rows.append(row)
            pats += p
            if curve is not None and key[3] == "valid":
                curves[key[:3]] = curve
            print(f"  {key}: F1 {row.get('f1_all', np.nan):.2f}, AP {row.get('ap_all', np.nan):.2f}")
        del meta

    runs = pd.DataFrame(rows)
    runs["config"] = pd.Categorical(runs["config"], CONFIGS, ordered=True)
    runs = runs.sort_values(["dataset", "variant", "config", "seed"]).reset_index(drop=True)
    runs["config"] = runs["config"].astype(str)
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)

    metric_cols = [c for c in SUMMARY_METRICS + THRESHOLD_METRICS +
                   ["summary_test_f1", "summary_test_ap", "val_f1", "epochs_run", "first_val_f1_pos_epoch",
                    "run_hours", "lambda_per_day"] if c in runs]
    agg = runs.groupby(["dataset", "config", "variant"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n", runs.groupby(["dataset", "config", "variant"]).size())
    summary = agg.reset_index()
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    tests = run_tests(runs, TEST_METRICS + [m for m in THRESHOLD_METRICS if m in runs])
    tests.to_csv(os.path.join(args.out, "tests.csv"), index=False)

    patterns = pd.DataFrame(pats)
    patterns.to_csv(os.path.join(args.out, "patterns.csv"), index=False)

    valid = runs[runs["valid"]]
    cost = valid.groupby(["dataset", "config"]).agg(
        n=("seed", "size"), params=("params", "mean"), epoch_train_s=("epoch_train_s", "mean"),
        epochs_run=("epochs_run", "mean"), run_hours=("run_hours", "mean"),
        peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean")).reset_index()
    base = cost[cost["config"] == "base"].set_index("dataset")
    cost["epoch_overhead_%"] = [100 * (r.epoch_train_s / base.loc[r.dataset, "epoch_train_s"] - 1)
                                if r.dataset in base.index else np.nan for r in cost.itertuples()]
    cost["params_extra"] = [r.params - base.loc[r.dataset, "params"] if r.dataset in base.index else np.nan
                            for r in cost.itertuples()]
    cost["config"] = pd.Categorical(cost["config"], CONFIGS, ordered=True)
    cost = cost.sort_values(["dataset", "config"]).reset_index(drop=True)
    cost["config"] = cost["config"].astype(str)
    cost.to_csv(os.path.join(args.out, "cost.csv"), index=False)

    plot_pr(curves, grid, args.out)
    write_report(runs, summary, tests, patterns, cost, args.out)
    print(f"\nwritten to {args.out}: runs.csv, summary.csv, tests.csv, patterns.csv, cost.csv, pr_*.png, laporan.md")


if __name__ == "__main__":
    main()
