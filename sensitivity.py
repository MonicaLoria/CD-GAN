"""
sensitivity.py -- Sections 15-20: legitimate, leak-free hyperparameter
sensitivity analysis.

Never touches the final TEST partition: every sweep here evaluates on the
internal VALIDATION partition produced by
``preprocessing.prepare_tuning_split``, with preprocessing fit ONLY on the
internal GAN-TRAIN partition. Configuration selection during sensitivity
testing uses this validation signal only -- never the final test set
(Section 18).

Staged design (Section 16) -- axes are swept ONE AT A TIME, never as a
full Cartesian product:

    Stage A: fix lambda_feature=1.0, fixed batch_size, sweep lambda_corr
             over a caller-supplied grid (manuscript default grid:
             [0.01, 0.05, 0.1, 0.5, 1.0]).
    Stage B: using the BEST lambda_corr from Stage A (by the configured
             selection score, never assumed), sweep lambda_feature over a
             caller-supplied grid (manuscript default: [0.5, 1.0, 2.0]).
    Stage C: (optional) compare batch_size values (manuscript default:
             64 vs 128), holding the Stage A/B winners fixed.

lambda_gp=10, n_critic=2, learning rates=2e-4, betas=(0.5, 0.9) are held
fixed throughout unless the caller explicitly overrides them via
``base_overrides`` -- this module never assumes any particular value
(e.g. lambda_corr=1.0) is "better" a priori (Section 20). The per-epoch
diagnostic columns already logged by trainer.py (``corr_weighted_loss``,
``corr_to_adv_ratio``) live in each run's ``training_history.csv`` under
its own output directory, so the adversarial-vs-covariance balance at
every grid point remains directly inspectable, not just the final
utility number.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common_eval import train_and_evaluate_one_configuration
from preprocessing import prepare_tuning_split, save_config_snapshot

DEFAULT_LAMBDA_CORR_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]
DEFAULT_LAMBDA_FEATURE_GRID = [0.5, 1.0, 2.0]
DEFAULT_BATCH_SIZE_GRID = [64, 128]
# Rev 4 (reviewer points 11-12): regularizer schedules and GP weight.
# gp_every = apply the gradient penalty every N critic steps;
# corr_every = apply L_dep every N generator steps. Each row of the output
# table carries training_time_sec, so the accuracy/cost trade-off that
# justifies the chosen interval is reported, not asserted.
DEFAULT_GP_EVERY_GRID = [1, 2, 5, 10]
DEFAULT_CORR_EVERY_GRID = [1, 2, 5, 10]
DEFAULT_LAMBDA_GP_GRID = [1.0, 5.0, 10.0]


def default_selection_score(summary_row: Dict[str, float]) -> float:
    """Section 17: a legitimate, documented, CONFIGURABLE selection score
    -- never a silently-weighted combination. This default is an EXPLICIT,
    equally-weighted average of mean F1, mean ROC-AUC, and mean PR-AUC on
    the internal VALIDATION partition:

        score = (f1 + roc_auc + pr_auc) / 3

    Callers who want a different score (e.g. one that also rewards
    fidelity, or penalizes the Section 16 corr/adv imbalance) pass their
    own ``selection_score_fn`` to ``run_sensitivity_stage`` /
    ``run_staged_sensitivity`` -- this module never hides a weighting
    choice inside opaque logic; whichever function is used is named
    explicitly in the printed output and recorded in the saved CSV
    (the full row -- including every fidelity column -- is always
    written out regardless of which scalar the score happens to use).
    """
    return float(np.mean([summary_row["f1"], summary_row["roc_auc"], summary_row["pr_auc"]]))


def _summarize_result(label: str, axis_name: str, axis_value, result) -> Dict:
    u = result.utility_df
    row = {
        "label": label,
        "axis": axis_name,
        "value": axis_value,
        "f1": float(u["f1"].mean()),
        "roc_auc": float(u["roc_auc"].mean()),
        "pr_auc": float(u["pr_auc"].mean()),
        "training_time_sec": result.training_time_sec,
        "seconds_per_epoch": result.training_time_sec / max(int(getattr(result.config, "epochs", 1)), 1),
        "synthetic_generation_time_sec": result.synthetic_generation_time_sec,
    }
    row.update(result.fidelity)   # wasserstein_normalized_mean, categorical_tv_complement_mean, pairwise_mae, density, coverage
    return row


def run_sensitivity_stage(
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
    axis_name: str,
    axis_values: Sequence,
    results_root: str,
    selection_score_fn: Callable[[Dict[str, float]], float] = default_selection_score,
) -> pd.DataFrame:
    """Runs ONE sweep axis (e.g. axis_name='lambda_corr',
    axis_values=[0.01, 0.05, ...]), holding every other CDGANConfig field
    at ``base_overrides``, all on the SAME leak-free split (fit ONCE for
    this seed, reused across every grid point -- Section 23: identical raw
    split reused across variants at a given seed). Evaluates on the
    internal VALIDATION partition only. Returns a DataFrame with one row
    per grid point, including the configured ``selection_score_fn``
    output, sorted best-first.
    """
    split = prepare_tuning_split(
        df, target_column, categorical_cols, numerical_cols, dataset_name,
        seed=seed, test_size=test_size, validation_size=validation_size, target_mapping=target_mapping,
    )
    fp = split["preprocessor"]
    real_fidelity_df = split["gan_train_df"]  # Section 4: fidelity is train-vs-synthetic

    out_root = os.path.join(results_root, dataset_name, "sensitivity", axis_name, f"seed{seed}")
    os.makedirs(out_root, exist_ok=True)
    save_config_snapshot(fp, os.path.join(out_root, "preprocessor_summary.json"))

    rows = []
    for value in axis_values:
        label = f"{axis_name}={value}"
        overrides = dict(base_overrides)
        overrides[axis_name] = value
        out_dir = os.path.join(out_root, str(value).replace(".", "p"))
        print(f"\n---- sensitivity[{axis_name}={value}] seed={seed} ----")
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
            config_overrides=overrides,
        )
        result.utility_df.to_csv(os.path.join(out_dir, "utility_on_validation.csv"), index=False)
        row = _summarize_result(label, axis_name, value, result)
        row["selection_score"] = selection_score_fn(row)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    table_path = os.path.join(out_root, f"sensitivity_{axis_name}_table.csv")
    table.to_csv(table_path, index=False)

    print(f"\n=== SENSITIVITY SWEEP: axis='{axis_name}' (internal VALIDATION partition only -- TEST untouched) ===")
    print(table.to_string(index=False))
    print(f"selection_score_fn = {selection_score_fn.__name__}")
    print(
        f"Best {axis_name} by selection_score: {table.iloc[0]['value']} "
        f"(selection_score={table.iloc[0]['selection_score']:.4f})"
    )
    print(f"Full table written to {table_path}\n")

    return table


def run_staged_sensitivity(
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
    results_root: str,
    lambda_corr_grid: Optional[Sequence[float]] = None,
    lambda_feature_grid: Optional[Sequence[float]] = None,
    batch_size_grid: Optional[Sequence[int]] = None,
    run_stage_c: bool = False,
    selection_score_fn: Callable[[Dict[str, float]], float] = default_selection_score,
) -> Dict[str, pd.DataFrame]:
    """Section 16: the full staged sweep (Stage A -> Stage B -> optional
    Stage C), never a Cartesian product. Returns a dict of the per-stage
    tables, keyed 'stage_a_lambda_corr' / 'stage_b_lambda_feature' /
    (optionally) 'stage_c_batch_size'.
    """
    lambda_corr_grid = list(lambda_corr_grid) if lambda_corr_grid else DEFAULT_LAMBDA_CORR_GRID
    lambda_feature_grid = list(lambda_feature_grid) if lambda_feature_grid else DEFAULT_LAMBDA_FEATURE_GRID
    batch_size_grid = list(batch_size_grid) if batch_size_grid else DEFAULT_BATCH_SIZE_GRID

    stage_a_base = dict(base_overrides)
    stage_a_base.setdefault("lambda_feature", 1.0)
    print("\n########## STAGE A: sweep lambda_corr (lambda_feature fixed) ##########")
    table_a = run_sensitivity_stage(
        dataset_name, df, target_column, categorical_cols, numerical_cols, target_mapping,
        seed, test_size, validation_size, stage_a_base, "lambda_corr", lambda_corr_grid,
        results_root, selection_score_fn,
    )
    best_lambda_corr = table_a.iloc[0]["value"]

    stage_b_base = dict(stage_a_base)
    stage_b_base["lambda_corr"] = best_lambda_corr
    print(f"\n########## STAGE B: sweep lambda_feature (lambda_corr fixed at best={best_lambda_corr}) ##########")
    table_b = run_sensitivity_stage(
        dataset_name, df, target_column, categorical_cols, numerical_cols, target_mapping,
        seed, test_size, validation_size, stage_b_base, "lambda_feature", lambda_feature_grid,
        results_root, selection_score_fn,
    )
    best_lambda_feature = table_b.iloc[0]["value"]

    results = {"stage_a_lambda_corr": table_a, "stage_b_lambda_feature": table_b}

    if run_stage_c:
        stage_c_base = dict(stage_b_base)
        stage_c_base["lambda_feature"] = best_lambda_feature
        print(
            f"\n########## STAGE C: compare batch_size "
            f"(lambda_corr={best_lambda_corr}, lambda_feature={best_lambda_feature}) ##########"
        )
        table_c = run_sensitivity_stage(
            dataset_name, df, target_column, categorical_cols, numerical_cols, target_mapping,
            seed, test_size, validation_size, stage_c_base, "batch_size", batch_size_grid,
            results_root, selection_score_fn,
        )
        results["stage_c_batch_size"] = table_c

    return results
