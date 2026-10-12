"""
regression_check.py -- Section 14: performance regression guard.

Compares TWO configurations (a "reference" and a "final") trained on the
IDENTICAL dataset / split / seed / preprocessing / epochs / evaluation
protocol, and prints a Metric / Reference / Final / Difference table. On
detecting a regression (any higher-is-better metric decreasing, or any
lower-is-better metric increasing, by more than ``tolerance`` relative to
the reference value) it prints:

    PERFORMANCE REGRESSION DETECTED -- INVESTIGATE BEFORE PRODUCTION

This tool NEVER auto-reverts anything -- it only reports. Per the
"IMPORTANT PHILOSOPHY" instruction this revision was built under: a
scientifically correct change that regresses a metric must be reported and
diagnosed, never silently reverted or tuned away just to make the number
go back up.

Both configurations are evaluated on the internal VALIDATION partition
produced by ``preprocessing.prepare_tuning_split`` -- the final TEST
partition it also produces is never touched by this module.

    python run_experiment.py --mode regression_check --dataset SICK \\
        --data-path dataset_38_sick.arff --seeds 0 --epochs 200 \\
        --reference-lambda-corr 0.05 --final-lambda-corr 1.0
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_eval import train_and_evaluate_one_configuration
from preprocessing import prepare_tuning_split, save_config_snapshot

# Section 4's fidelity metrics + Section 11's utility metrics, with each
# metric's direction made explicit (never implicit) so "regression" always
# means "got worse", regardless of whether the metric is higher-is-better
# or lower-is-better.
HIGHER_IS_BETTER = ["f1", "roc_auc", "pr_auc", "categorical_tv_complement_mean", "coverage", "density"]
LOWER_IS_BETTER = ["wasserstein_normalized_mean", "pairwise_mae"]


def _summarize_utility(utility_df: pd.DataFrame) -> Dict[str, float]:
    """Mean across classifiers -- one scalar per utility metric for the
    regression-check table. The full per-classifier utility_df is also
    written to disk (utility_on_validation.csv) for closer inspection."""
    return {
        "f1": float(utility_df["f1"].mean()),
        "roc_auc": float(utility_df["roc_auc"].mean()),
        "pr_auc": float(utility_df["pr_auc"].mean()),
    }


def run_regression_check(
    dataset_name: str,
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_mapping: Optional[Dict],
    seed: int,
    test_size: float,
    validation_size: float,
    base_overrides: Dict,
    reference_overrides: Dict,
    final_overrides: Dict,
    results_root: str,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Trains 'reference' and 'final' configurations on the IDENTICAL
    split (same seed, same preprocessing -- fit ONCE and reused for both
    arms) and returns the Metric/Reference/Final/Difference table. Both
    configurations are evaluated on the internal VALIDATION partition; the
    TEST partition produced by ``prepare_tuning_split`` is not touched by
    this function at all.
    """
    split = prepare_tuning_split(
        df, target_column, categorical_cols, numerical_cols, dataset_name,
        seed=seed, test_size=test_size, validation_size=validation_size, target_mapping=target_mapping,
    )
    fp = split["preprocessor"]
    # Section 4: fidelity is real TRAIN vs synthetic -- the GAN-TRAIN
    # partition here, never validation or test.
    real_fidelity_df = split["gan_train_df"]

    out_root = os.path.join(results_root, dataset_name, "regression_check", f"seed{seed}")
    os.makedirs(out_root, exist_ok=True)
    save_config_snapshot(fp, os.path.join(out_root, "preprocessor_summary.json"))

    results = {}
    for label, overrides in [("reference", reference_overrides), ("final", final_overrides)]:
        merged = dict(base_overrides)
        merged.update(overrides)
        out_dir = os.path.join(out_root, label)
        result = train_and_evaluate_one_configuration(
            label=label,
            dataset_name=dataset_name,
            seed=seed,
            X_train=split["X_gan_train"], y_train=split["y_gan_train"],
            X_eval=split["X_validation"], y_eval=split["y_validation"],
            fp=fp,
            real_fidelity_df=real_fidelity_df,
            numerical_cols=numerical_cols, categorical_cols=categorical_cols,
            out_dir=out_dir,
            config_overrides=merged,
        )
        results[label] = result
        result.utility_df.to_csv(os.path.join(out_dir, "utility_on_validation.csv"), index=False)
        print(f"[regression_check:{label}] config: {merged}")

    ref_metrics = {**_summarize_utility(results["reference"].utility_df), **results["reference"].fidelity}
    final_metrics = {**_summarize_utility(results["final"].utility_df), **results["final"].fidelity}

    rows = []
    regression_detected = False
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        ref_v = ref_metrics.get(metric, float("nan"))
        final_v = final_metrics.get(metric, float("nan"))
        diff = final_v - ref_v
        direction = "higher_better" if metric in HIGHER_IS_BETTER else "lower_better"
        is_regression = False
        if not (np.isnan(ref_v) or np.isnan(final_v)):
            denom = max(abs(ref_v), 1e-8)
            rel_change = diff / denom
            if direction == "higher_better" and rel_change < -tolerance:
                is_regression = True
            if direction == "lower_better" and rel_change > tolerance:
                is_regression = True
        regression_detected = regression_detected or is_regression
        rows.append({
            "metric": metric,
            "direction": direction,
            "reference": ref_v,
            "final": final_v,
            "difference": diff,
            "regression_flag": is_regression,
        })

    table = pd.DataFrame(rows)
    table_path = os.path.join(out_root, "regression_check_table.csv")
    table.to_csv(table_path, index=False)

    print("\n=== PERFORMANCE REGRESSION CHECK (internal VALIDATION partition only -- TEST untouched) ===")
    print(table.to_string(index=False))
    print(f"tolerance = {tolerance:.2%} relative degradation")
    if regression_detected:
        print("\nPERFORMANCE REGRESSION DETECTED -- INVESTIGATE BEFORE PRODUCTION")
    else:
        print("\nNo regression beyond tolerance detected on the metrics above.")
    print(f"Full table written to {table_path}\n")

    return table
