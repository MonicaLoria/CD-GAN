"""
make_tables.py -- Rev 4: build every paper table from ONE set of run
folders, with consistency checks (reviewer points 3, 8, 13, 15, 16, 17, 19).

Torch-free. Expected layout (written by run_experiment.py and
run_baselines.py in Rev 4):

    <results_root>/<DATASET>/<run_name>/seed<k>/
        quality_summary.csv      (quality_metrics.evaluate_synthetic)
        utility.csv
        split_fingerprint.json   (hash of the real train/test row split)
        runtime.json             (optional)
        param_counts.json        (optional, CD-GAN only)
        run_config.json          (hyperparameters actually used)

``run_name`` is "full" for CD-GAN, an ablation name (wo_gp, wo_corr,
wo_dual, pearson_corr, ...) for CD-GAN variants, and "baseline_<METHOD>"
for baselines.

Outputs (in --out-dir):
    all_runs_long.csv          every run, one row, every scalar metric
    table_main_utility.csv     CD-GAN + baselines, mean +/- SD per dataset
    table_fidelity.csv         type-aware fidelity + dependency + detection
    table_diversity.csv        mode-collapse / diversity metrics
    table_minority.csv         minority-class utility and fidelity
    table_privacy.csv          DCR / NNDR / copy ratios
    table_cost.csv             training time, s/epoch, generation time,
                               peak GPU memory, parameters
    table_ablation_absolute.csv  CD-GAN variants, ABSOLUTE mean +/- SD
    table_ablation_delta.csv     variant minus full, mean over seeds
                                 (a separate table -- never mixed with
                                 absolute scores)
    consistency_report.txt     all checks and their outcome
    stats/                     stats_analysis.py output for main metrics

The script stops with a non-zero exit code if a consistency check fails,
unless --allow-inconsistent is passed (then the failures are written at
the top of consistency_report.txt).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

import pandas as pd

import stats_analysis

MAIN_UTILITY = ["util_f1", "util_roc_auc", "util_pr_auc", "util_mcc", "util_f1_macro", "util_balanced_accuracy"]
FIDELITY = ["num_wd_norm_mean", "num_ks_stat_mean", "cat_tv_distance_mean", "cat_jsd_mean",
            "assoc_mae", "assoc_mae_class_conditional", "num_pearson_mae",
            "detection_auc_lr", "detection_auc_gbm"]
DIVERSITY = ["syn_unique_row_ratio", "syn_duplicate_ratio", "category_coverage", "rare_category_coverage",
             "nn_diversity_ratio", "density", "coverage"]
MINORITY = ["util_minority_precision", "util_minority_recall", "util_minority_f1", "util_minority_pr_auc",
            "minority_num_wd_norm_mean", "minority_cat_tv_distance_mean", "minority_assoc_mae",
            "minority_coverage"]
PRIVACY = ["dcr_mean", "dcr_median", "nndr_mean", "exact_copy_ratio", "near_copy_ratio"]
COST = ["training_time_sec", "seconds_per_epoch", "synthetic_generation_time_sec",
        "generation_sec_per_10k_rows", "peak_cuda_memory_mb", "total_parameters"]


def method_label(run_name: str) -> str:
    if run_name == "full":
        return "CD-GAN"
    if run_name.startswith("baseline_"):
        return run_name[len("baseline_"):]
    return f"CD-GAN[{run_name}]"


def collect(results_root: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_root, "*", "*", "seed*", "quality_summary.csv"))):
        seed_dir = os.path.dirname(path)
        run_dir = os.path.dirname(seed_dir)
        ds = os.path.basename(os.path.dirname(run_dir))
        run_name = os.path.basename(run_dir)
        row = pd.read_csv(path).iloc[0].to_dict()
        row.update({"dataset": ds, "run_name": run_name, "method": method_label(run_name),
                    "seed": int(os.path.basename(seed_dir)[4:]), "run_dir": seed_dir})
        for fname in ["runtime.json", "param_counts.json", "split_fingerprint.json"]:
            fp = os.path.join(seed_dir, fname)
            if os.path.exists(fp):
                with open(fp) as f:
                    for k, v in json.load(f).items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            row[k] = v
        if "training_time_sec" in row and row.get("epochs"):
            row.setdefault("seconds_per_epoch", row["training_time_sec"] / max(float(row["epochs"]), 1.0))
        if "synthetic_generation_time_sec" in row and row.get("n_synthetic"):
            row["generation_sec_per_10k_rows"] = row["synthetic_generation_time_sec"] / row["n_synthetic"] * 1e4
        rows.append(row)
    if not rows:
        raise SystemExit(f"No quality_summary.csv found under {results_root}/<DATASET>/<run>/seed*/")
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame, metrics: List[str], methods: List[str], decimals: int = 4) -> pd.DataFrame:
    sub = df[df["method"].isin(methods)]
    out = []
    for (ds, m), g in sub.groupby(["dataset", "method"]):
        rec = {"dataset": ds, "method": m, "n_seeds": int(g["seed"].nunique())}
        for k in metrics:
            if k not in g.columns:
                continue
            v = pd.to_numeric(g[k], errors="coerce").dropna()
            if len(v) == 0:
                rec[k] = ""
            elif len(v) == 1:
                rec[k] = f"{v.iloc[0]:.{decimals}f} (n=1)"
            else:
                rec[k] = f"{v.mean():.{decimals}f} +/- {v.std(ddof=1):.{decimals}f}"
        out.append(rec)
    return pd.DataFrame(out).sort_values(["dataset", "method"])


def consistency_checks(df: pd.DataFrame) -> List[str]:
    problems = []
    # 1. every method on a (dataset, seed) must share the same real split
    if "split_sha1" in df.columns:
        for (ds, seed), g in df.groupby(["dataset", "seed"]):
            vals = g["split_sha1"].dropna().unique()
            if len(vals) > 1:
                bad = g.groupby("split_sha1")["method"].apply(list).to_dict()
                problems.append(f"[split] {ds} seed {seed}: methods used different splits: {bad}")
    else:
        problems.append("[split] no split_fingerprint.json found; cannot prove identical splits")
    # 2. seeds: every method should have the same seed set per dataset
    for ds, g in df.groupby("dataset"):
        seed_sets = g.groupby("method")["seed"].apply(lambda s: tuple(sorted(s))).to_dict()
        if len(set(seed_sets.values())) > 1:
            problems.append(f"[seeds] {ds}: methods have different seed sets: {seed_sets}")
    # 3. ablation 'full' == main-table CD-GAN (they are the same run folders
    #    by construction; this guards against copying numbers by hand)
    full = df[df["run_name"] == "full"]
    for (ds, seed), g in full.groupby(["dataset", "seed"]):
        if len(g) != 1:
            problems.append(f"[full] {ds} seed {seed}: {len(g)} 'full' runs found (expected 1)")
    # 4. hyperparameters identical across seeds for each run
    for (ds, run), g in df.groupby(["dataset", "run_name"]):
        cfgs = set()
        for d in g["run_dir"]:
            p = os.path.join(d, "run_config.json")
            if os.path.exists(p):
                with open(p) as f:
                    c = json.load(f)
                for k in ("seed", "resume", "generated_counts", "n_synthetic", "versions"):
                    c.pop(k, None)
                if isinstance(c.get("hyperparameters"), dict):
                    c["hyperparameters"].pop("source_file", None)
                cfgs.add(json.dumps(c, sort_keys=True, default=str))
        if len(cfgs) > 1:
            problems.append(f"[config] {ds}/{run}: hyperparameters differ across seeds")
    return problems


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ablation-runs", nargs="*", default=None,
                    help="run names to include in the ablation table (default: every non-baseline run)")
    ap.add_argument("--allow-inconsistent", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    df = collect(args.results_root)
    df.to_csv(os.path.join(args.out_dir, "all_runs_long.csv"), index=False)

    problems = consistency_checks(df)
    report = ["CONSISTENCY CHECKS", "=" * 60]
    report += (problems or ["All checks passed."])
    report.append("")
    report.append("Utility protocol per method (TSTR = synthetic only; 'resampled' = real + generated rows):")
    if "utility_protocol" in df.columns:
        report += [f"  {m}: {p}" for m, p in df.groupby("method")["utility_protocol"].agg(
            lambda s: ", ".join(sorted(set(map(str, s))))).items()]
    with open(os.path.join(args.out_dir, "consistency_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    if problems and not args.allow_inconsistent:
        sys.exit("Consistency checks failed -- tables NOT written. Fix the runs or pass --allow-inconsistent.")

    main_methods = sorted(m for m in df["method"].unique() if not m.startswith("CD-GAN["))
    mean_sd(df, MAIN_UTILITY, main_methods).to_csv(os.path.join(args.out_dir, "table_main_utility.csv"), index=False)
    mean_sd(df, FIDELITY, main_methods).to_csv(os.path.join(args.out_dir, "table_fidelity.csv"), index=False)
    mean_sd(df, DIVERSITY, main_methods).to_csv(os.path.join(args.out_dir, "table_diversity.csv"), index=False)
    mean_sd(df, MINORITY, main_methods).to_csv(os.path.join(args.out_dir, "table_minority.csv"), index=False)
    mean_sd(df, PRIVACY, main_methods).to_csv(os.path.join(args.out_dir, "table_privacy.csv"), index=False)
    mean_sd(df, COST, sorted(df["method"].unique()), decimals=2).to_csv(
        os.path.join(args.out_dir, "table_cost.csv"), index=False)

    abl_runs = args.ablation_runs or sorted(r for r in df["run_name"].unique() if not r.startswith("baseline_"))
    abl = df[df["run_name"].isin(abl_runs)]
    if not abl.empty:
        abl_methods = sorted(abl["method"].unique())
        abl_metrics = ["util_f1", "util_pr_auc", "util_roc_auc", "util_minority_recall",
                       "num_wd_norm_mean", "cat_tv_distance_mean", "assoc_mae"]
        mean_sd(abl, abl_metrics, abl_methods).to_csv(
            os.path.join(args.out_dir, "table_ablation_absolute.csv"), index=False)
        # delta table: paired by (dataset, seed), variant minus full
        base = abl[abl["run_name"] == "full"].set_index(["dataset", "seed"])
        deltas = []
        for run, g in abl[abl["run_name"] != "full"].groupby("run_name"):
            g = g.set_index(["dataset", "seed"])
            common = g.index.intersection(base.index)
            for ds in sorted(set(i[0] for i in common)):
                idx = [i for i in common if i[0] == ds]
                rec = {"dataset": ds, "variant": method_label(run), "n_seeds": len(idx)}
                for k in abl_metrics:
                    if k in g.columns:
                        d = pd.to_numeric(g.loc[idx, k]) - pd.to_numeric(base.loc[idx, k])
                        rec[f"delta_{k}"] = round(float(d.mean()), 4)
                deltas.append(rec)
        pd.DataFrame(deltas).to_csv(os.path.join(args.out_dir, "table_ablation_delta.csv"), index=False)

    # long file for stats_analysis (main comparison only)
    # REAL (TRTR) is a reference row in the tables, never a competitor in the tests
    test_methods = [m for m in main_methods if m != "REAL"]
    long = df[df["method"].isin(test_methods)][["method", "dataset", "seed"] +
                                                 [c for c in MAIN_UTILITY + FIDELITY if c in df.columns]]
    long_path = os.path.join(args.out_dir, "stats_input_long.csv")
    long.to_csv(long_path, index=False)
    if df["dataset"].nunique() >= 2 and len(test_methods) >= 2 and "CD-GAN" in test_methods:
        metrics = [c for c in ["util_f1", "util_roc_auc", "util_pr_auc", "util_mcc",
                               "num_wd_norm_mean", "cat_tv_distance_mean", "assoc_mae"] if c in long.columns]
        stats_analysis.main(["--input", long_path, "--out-dir", os.path.join(args.out_dir, "stats"),
                             "--metrics", *metrics, "--control", "CD-GAN"])
    print(f"\nTables written to {args.out_dir}")


if __name__ == "__main__":
    main()
