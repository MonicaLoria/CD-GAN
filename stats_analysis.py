"""
stats_analysis.py -- Rev 4: statistical comparison of CD-GAN against
baselines across datasets (reviewer points 6, 15 and 18).

Torch-free. Reads one long-format CSV (one row per method x dataset x seed
[x classifier]) and writes, for EVERY metric separately:

  * mean_sd_<metric>.csv        -- method x dataset table, "mean +/- SD"
                                   across seeds (classifiers averaged first
                                   within each seed)
  * friedman_<metric>.csv       -- Friedman chi2, df, p; Iman-Davenport F,
                                   df1, df2, p; k, N; Nemenyi q_alpha and
                                   critical difference (CD)
  * avg_ranks_<metric>.csv      -- average Friedman rank per method
  * nemenyi_pvalues_<metric>.csv-- all-pairs Nemenyi p-value matrix
  * control_tests_<metric>.csv  -- CD-GAN (control) vs every other method:
                                   Wilcoxon signed-rank over datasets,
                                   Holm-adjusted p, matched-pairs
                                   rank-biserial effect size, median paired
                                   difference with bootstrap 95% CI,
                                   win/tie/loss counts, rank gap vs CD and
                                   Nemenyi significance
  * paired_seed_<metric>.csv    -- per dataset: CD-GAN minus each method,
                                   paired by seed, mean difference, 95% t CI
                                   and Cohen's d_z (only where both methods
                                   share >= 2 seeds)
  * subgroup_ranks_<metric>.csv -- Friedman ranks recomputed inside each
                                   method family (e.g. oversampling vs deep
                                   generative), always including the control
  * summary.md                  -- one paragraph per metric with the exact
                                   numbers, ready to quote in the paper

Directions: metrics whose name contains any of LOWER_IS_BETTER_KEYS (e.g.
wasserstein, ks_stat, tv_distance, jsd, mae, detection_auc_gap) are ranked
ascending; all others descending. Override with --lower-is-better.

Usage
-----
    python stats_analysis.py --input results_long.csv --out-dir stats \
        --metrics f1 roc_auc pr_auc wasserstein_normalized_mean \
        --control CD-GAN

The input must contain columns: method, dataset, seed, and each metric.
An optional 'classifier' column is averaged within (method, dataset, seed)
before any test. Methods missing a dataset are dropped from the Friedman
analysis for that metric (reported, never silently imputed).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

LOWER_IS_BETTER_KEYS = (
    "wasserstein", "wd_", "ks_stat", "tv_distance", "jsd", "kld", "mae", "rmse", "frobenius",
    "detection_auc_gap", "pmse", "duplicate", "copy_ratio", "time_sec", "memory",
)

DEFAULT_FAMILIES = {
    "SMOTE": "oversampling", "ADASYN": "oversampling", "ROS": "oversampling",
    "BorderlineSMOTE": "oversampling", "SMOTENC": "oversampling",
    "CTGAN": "deep_generative", "TVAE": "deep_generative", "CopulaGAN": "deep_generative",
    "CTABGAN+": "deep_generative", "TabDDPM": "deep_generative", "TWAE": "deep_generative",
    "TABULA": "deep_generative", "TTVAE": "deep_generative",
    "GaussianCopula": "statistical", "Copula": "statistical",
}


# ----------------------------------------------------------------------
def is_lower_better(metric: str, override: Sequence[str] = ()) -> bool:
    if metric in override:
        return True
    m = metric.lower()
    return any(k in m for k in LOWER_IS_BETTER_KEYS)


def seed_level(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average classifiers within (method, dataset, seed)."""
    keys = ["method", "dataset", "seed"]
    return df.groupby(keys, as_index=False)[metric].mean()


def dataset_level(seed_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Method x dataset matrix of means across seeds."""
    return seed_df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")


def mean_sd_table(seed_df: pd.DataFrame, metric: str, decimals: int = 4) -> pd.DataFrame:
    g = seed_df.groupby(["method", "dataset"])[metric]
    mean = g.mean().unstack("dataset")
    sd = g.std(ddof=1).unstack("dataset")
    n = g.count().unstack("dataset")
    out = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            m, s, k = mean.loc[r, c], sd.loc[r, c], n.loc[r, c]
            if pd.isna(m):
                out.loc[r, c] = ""
            elif pd.isna(s) or k < 2:
                out.loc[r, c] = f"{m:.{decimals}f} (n={int(k)})"
            else:
                out.loc[r, c] = f"{m:.{decimals}f} +/- {s:.{decimals}f}"
    out["mean_over_datasets"] = mean.mean(axis=1).round(decimals)
    return out


def friedman_ranks(mat: pd.DataFrame, lower_better: bool) -> pd.DataFrame:
    """Rank methods within each dataset (row); 1 = best; ties averaged."""
    vals = mat.to_numpy(dtype=float)
    signed = vals if lower_better else -vals
    ranks = np.vstack([stats.rankdata(row) for row in signed])
    return pd.DataFrame(ranks, index=mat.index, columns=mat.columns)


def nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Critical value q_alpha of the Nemenyi test (Demsar 2006), i.e. the
    studentized range quantile divided by sqrt(2)."""
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def friedman_test(mat: pd.DataFrame, lower_better: bool, alpha: float = 0.05) -> Dict[str, float]:
    N, k = mat.shape
    ranks = friedman_ranks(mat, lower_better)
    R = ranks.mean(axis=0).to_numpy()
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    df = k - 1
    p_chi2 = float(stats.chi2.sf(chi2, df))
    denom = N * (k - 1) - chi2
    if denom <= 0:
        F, p_F = float("inf"), 0.0
    else:
        F = (N - 1) * chi2 / denom
        p_F = float(stats.f.sf(F, k - 1, (k - 1) * (N - 1)))
    q = nemenyi_q_alpha(k, alpha)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return {
        "k_methods": k, "N_datasets": N,
        "friedman_chi2": float(chi2), "friedman_df": df, "friedman_p": p_chi2,
        "iman_davenport_F": float(F), "iman_davenport_df1": k - 1,
        "iman_davenport_df2": (k - 1) * (N - 1), "iman_davenport_p": p_F,
        "alpha": alpha, "nemenyi_q_alpha": q, "nemenyi_CD": float(cd),
    }


def nemenyi_pvalues(avg_ranks: pd.Series, N: int) -> pd.DataFrame:
    """All-pairs Nemenyi p-values from average ranks."""
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6 * N))
    names = list(avg_ranks.index)
    P = pd.DataFrame(np.ones((k, k)), index=names, columns=names)
    for i in range(k):
        for j in range(k):
            if i != j:
                s = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j]) / se * np.sqrt(2)
                P.iloc[i, j] = float(stats.studentized_range.sf(s, k, np.inf))
    return P


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def bootstrap_ci_median(d: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    if len(d) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def control_tests(mat: pd.DataFrame, control: str, lower_better: bool, ranks: pd.DataFrame,
                  cd: float, n_boot: int = 10000) -> pd.DataFrame:
    """CD-GAN vs each other method over datasets (dataset means)."""
    rows = []
    sign = -1.0 if lower_better else 1.0  # positive diff == control better
    avg = ranks.mean(axis=0)
    for m in mat.columns:
        if m == control:
            continue
        d = sign * (mat[control].to_numpy() - mat[m].to_numpy())
        wins, ties, losses = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
        nz = d[d != 0]
        if len(nz) == 0:
            W_stat, p = float("nan"), 1.0
            rbc = 0.0
        else:
            res = stats.wilcoxon(nz, alternative="two-sided")
            W_stat, p = float(res.statistic), float(res.pvalue)
            r = stats.rankdata(np.abs(nz))
            w_pos, w_neg = r[nz > 0].sum(), r[nz < 0].sum()
            rbc = float((w_pos - w_neg) / (w_pos + w_neg))
        lo, hi = bootstrap_ci_median(d, n_boot=n_boot)
        gap = float(avg[m] - avg[control])
        rows.append({
            "control": control, "method": m,
            "wins": wins, "ties": ties, "losses": losses,
            "median_diff_control_better_positive": float(np.median(d)),
            "median_diff_ci95_low": lo, "median_diff_ci95_high": hi,
            "wilcoxon_W": W_stat, "wilcoxon_p": p,
            "rank_biserial_r": rbc,
            "avg_rank_control": float(avg[control]), "avg_rank_method": float(avg[m]),
            "rank_gap": gap, "nemenyi_significant": bool(gap > cd),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].tolist())
        out["wilcoxon_significant_holm_0.05"] = out["wilcoxon_p_holm"] < 0.05
    return out


def paired_seed_tests(seed_df: pd.DataFrame, metric: str, control: str, lower_better: bool) -> pd.DataFrame:
    """Per dataset, control minus method paired by seed (only shared seeds)."""
    sign = -1.0 if lower_better else 1.0
    rows = []
    for ds, sub in seed_df.groupby("dataset"):
        piv = sub.pivot_table(index="seed", columns="method", values=metric)
        if control not in piv.columns:
            continue
        for m in piv.columns:
            if m == control:
                continue
            pair = piv[[control, m]].dropna()
            n = len(pair)
            if n < 2:
                continue
            d = sign * (pair[control] - pair[m]).to_numpy()
            mean, sd = float(d.mean()), float(d.std(ddof=1))
            if sd > 0:
                t = stats.t.ppf(0.975, n - 1)
                lo, hi = mean - t * sd / np.sqrt(n), mean + t * sd / np.sqrt(n)
                dz = mean / sd
            else:
                lo = hi = mean
                dz = float("inf") if mean != 0 else 0.0
            rows.append({
                "dataset": ds, "control": control, "method": m, "n_seeds": n,
                "mean_diff_control_better_positive": mean,
                "ci95_low": float(lo), "ci95_high": float(hi), "cohens_dz": float(dz),
            })
    return pd.DataFrame(rows)


def subgroup_ranks(mat: pd.DataFrame, control: str, lower_better: bool,
                   families: Dict[str, str]) -> pd.DataFrame:
    rows = []
    fam_of = {m: families.get(m, "other") for m in mat.columns if m != control}
    for fam in sorted(set(fam_of.values())):
        members = [control] + [m for m, f in fam_of.items() if f == fam]
        if len(members) < 2:
            continue
        sub = mat[members]
        ranks = friedman_ranks(sub, lower_better).mean(axis=0)
        for m, r in ranks.items():
            rows.append({"family": fam, "method": m, "avg_rank_within_family": float(r),
                         "k_in_family": len(members)})
    return pd.DataFrame(rows)


def cd_diagram(avg_ranks: pd.Series, cd: float, path: str, title: str) -> None:
    """Simple critical-difference diagram (Demsar 2006 style)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    r = avg_ranks.sort_values()
    k = len(r)
    fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1.6))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-k - 1, 2)
    ax.axhline(0, color="black", lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0, 0.15], color="black", lw=1)
        ax.text(i, 0.3, str(i), ha="center", fontsize=8)
    ax.plot([1, 1 + cd], [1.3, 1.3], color="black", lw=2)
    ax.text(1 + cd / 2, 1.5, f"CD = {cd:.3f}", ha="center", fontsize=8)
    for idx, (name, val) in enumerate(r.items()):
        y = -(idx + 1) * 0.8
        ax.plot([val, val], [0, y], color="grey", lw=0.8)
        ax.text(val + 0.05, y, f"{name} ({val:.2f})", va="center", fontsize=8)
    # cliques: maximal groups whose rank span < CD
    vals = r.to_numpy()
    y = -k * 0.8 - 0.4
    last_end = -1
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and j > last_end:
            ax.plot([vals[i], vals[j]], [y, y], color="red", lw=3)
            y -= 0.25
            last_end = j
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
def analyse_metric(df: pd.DataFrame, metric: str, control: str, out_dir: str, alpha: float,
                   lower_override: Sequence[str], families: Dict[str, str]) -> str:
    lower = is_lower_better(metric, lower_override)
    sdf = seed_level(df.dropna(subset=[metric]), metric)
    mean_sd_table(sdf, metric).to_csv(os.path.join(out_dir, f"mean_sd_{metric}.csv"))

    mat = dataset_level(sdf, metric)
    incomplete = [m for m in mat.columns if mat[m].isna().any()]
    mat = mat.drop(columns=incomplete)
    if control not in mat.columns:
        return f"### {metric}\nControl '{control}' missing on some dataset; tests skipped.\n"
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return f"### {metric}\nNot enough datasets/methods for Friedman analysis.\n"

    fr = friedman_test(mat, lower, alpha)
    pd.DataFrame([fr]).to_csv(os.path.join(out_dir, f"friedman_{metric}.csv"), index=False)
    ranks = friedman_ranks(mat, lower)
    avg = ranks.mean(axis=0).sort_values()
    avg.rename("avg_rank").to_csv(os.path.join(out_dir, f"avg_ranks_{metric}.csv"))
    nemenyi_pvalues(avg, mat.shape[0]).to_csv(os.path.join(out_dir, f"nemenyi_pvalues_{metric}.csv"))
    ct = control_tests(mat, control, lower, ranks, fr["nemenyi_CD"])
    ct.to_csv(os.path.join(out_dir, f"control_tests_{metric}.csv"), index=False)
    paired_seed_tests(sdf, metric, control, lower).to_csv(
        os.path.join(out_dir, f"paired_seed_{metric}.csv"), index=False)
    subgroup_ranks(mat, control, lower, families).to_csv(
        os.path.join(out_dir, f"subgroup_ranks_{metric}.csv"), index=False)
    cd_diagram(avg, fr["nemenyi_CD"], os.path.join(out_dir, f"cd_diagram_{metric}.png"),
               f"{metric} (k={fr['k_methods']}, N={fr['N_datasets']}, alpha={alpha})")

    not_sig_nem = ct.loc[~ct["nemenyi_significant"], "method"].tolist() if not ct.empty else []
    not_sig_wil = ct.loc[~ct["wilcoxon_significant_holm_0.05"], "method"].tolist() if not ct.empty else []
    lines = [
        f"### {metric} ({'lower' if lower else 'higher'} is better)",
        f"Friedman chi2({fr['friedman_df']}) = {fr['friedman_chi2']:.2f}, p = {fr['friedman_p']:.3g}; "
        f"Iman-Davenport F({fr['iman_davenport_df1']}, {fr['iman_davenport_df2']}) = "
        f"{fr['iman_davenport_F']:.2f}, p = {fr['iman_davenport_p']:.3g}; "
        f"k = {fr['k_methods']} methods, N = {fr['N_datasets']} datasets.",
        f"Nemenyi CD (alpha = {alpha}) = {fr['nemenyi_CD']:.3f}; {control} average rank = {avg[control]:.3f}.",
        f"Not significantly different from {control} by Nemenyi: {', '.join(not_sig_nem) or 'none'}.",
        f"Not significantly different from {control} by Wilcoxon-Holm: {', '.join(not_sig_wil) or 'none'}.",
    ]
    if incomplete:
        lines.append(f"Excluded (missing at least one dataset): {', '.join(incomplete)}.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="long-format CSV: method,dataset,seed[,classifier],<metrics>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--control", default="CD-GAN")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--lower-is-better", nargs="*", default=[],
                    help="metrics to rank ascending in addition to the automatic name-based rule")
    ap.add_argument("--family", nargs="*", default=[],
                    help="METHOD=family overrides, e.g. SMOTE=oversampling CTGAN=deep_generative")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input)
    for col in ["method", "dataset", "seed"]:
        if col not in df.columns:
            raise ValueError(f"input CSV must contain column '{col}'")
    families = dict(DEFAULT_FAMILIES)
    for item in args.family:
        k, v = item.split("=", 1)
        families[k] = v
    os.makedirs(args.out_dir, exist_ok=True)

    summary = ["# Statistical analysis summary", ""]
    for metric in args.metrics:
        if metric not in df.columns:
            summary.append(f"### {metric}\nColumn not found in input.\n")
            continue
        summary.append(analyse_metric(df, metric, args.control, args.out_dir, args.alpha,
                                      args.lower_is_better, families))
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
