"""
common_eval.py -- shared "train one CD-GAN configuration and evaluate it"
helper used by BOTH regression_check.py (Section 14) and sensitivity.py
(Sections 15-20), so the two tools can never silently diverge from each
other -- or from run_experiment.py's own single-seed path -- in how a
configuration is built, trained, and scored.

This module trains EXACTLY the same architecture as run_experiment.py's
``run_single_seed()``: it does not add, remove, or reweight any loss term,
and it does not implement its own copy of the CD-GAN model/training loop.
It differs from ``run_single_seed()`` only in WHICH already-computed
(X_train, y_train, X_eval, y_eval, real_fidelity_df) arrays it is handed --
the caller controls whether the "downstream test set" is the internal
VALIDATION partition (tuning / regression-check, via
``preprocessing.prepare_tuning_split``) or the untouched final TEST
partition (production, via ``preprocessing.prepare_seed_split`` in
run_experiment.py).

Fidelity metrics (Section 4) are ALWAYS computed against a
TRAINING-partition raw dataframe (``real_fidelity_df``), regardless of
what ``X_eval``/``y_eval`` are being used for downstream utility -- so a
sensitivity sweep or regression check can never accidentally score
fidelity against validation or test data.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from evaluation import (
    build_tstr_training_set,
    categorical_quality_report,
    density_coverage,
    evaluate_tstr_per_classifier,
    pairwise_relationship_error,
    statistical_quality_report,
)
from model import build_models, count_parameters
from preprocessing import FittedPreprocessor, impute_dataframe, inverse_transform
from trainer import ABLATION_FLAGS, CDGANConfig, CDGANTrainer, set_all_seeds


@dataclass
class ConfigurationResult:
    label: str
    config: CDGANConfig
    utility_df: pd.DataFrame          # per classifier: f1/roc_auc/pr_auc
    fidelity: Dict[str, float]        # aggregated scalar fidelity metrics (train-vs-synthetic)
    training_time_sec: float
    synthetic_generation_time_sec: float


def _aggregate_fidelity(
    real_fidelity_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    fp: FittedPreprocessor,
    X_real_fidelity_encoded: np.ndarray,
    X_syn_proj: np.ndarray,
) -> Dict[str, float]:
    """Section 4: fidelity = real TRAIN vs synthetic, aggregated to one
    scalar per metric family (mean across columns) so it fits in a single
    row of a comparison/sensitivity table. The full per-column tables are
    still available via ``evaluation.statistical_quality_report`` /
    ``categorical_quality_report`` directly, if finer detail is needed."""
    stat_df = statistical_quality_report(
        real_fidelity_df, fake_df, numerical_cols, numerical_percentile_bounds=fp.numerical_percentile_bounds,
    )
    cat_categories_by_col = dict(zip(fp.categorical_cols, fp.categorical_categories))
    cat_df = categorical_quality_report(
        real_fidelity_df, fake_df, categorical_cols, categorical_categories=cat_categories_by_col,
    )
    pairwise = pairwise_relationship_error(
        real_fidelity_df, fake_df, numerical_cols, categorical_cols,
        binary_category_orders=fp.binary_category_orders,
    )
    dc = density_coverage(X_real_fidelity_encoded, X_syn_proj)

    wd_norm_mean = float(stat_df["wasserstein_normalized"].mean()) if not stat_df.empty else float("nan")
    tv_complement_mean = float(cat_df["tv_complement"].mean()) if not cat_df.empty else float("nan")

    return {
        "wasserstein_normalized_mean": wd_norm_mean,
        "categorical_tv_complement_mean": tv_complement_mean,
        "pairwise_mae": pairwise.get("pairwise_mae", float("nan")),
        "density": dc.get("density", float("nan")),
        "coverage": dc.get("coverage", float("nan")),
    }


def train_and_evaluate_one_configuration(
    label: str,
    dataset_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fp: FittedPreprocessor,
    real_fidelity_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    out_dir: str,
    config_overrides: Dict,
    evaluation_mode: str = "TSTR",
) -> ConfigurationResult:
    """Trains ONE CD-GAN configuration on (X_train, y_train) and evaluates
    downstream utility on (X_eval, y_eval) -- which the CALLER decides is
    either an internal VALIDATION partition (tuning/regression-check) or
    the final TEST partition (production) -- plus fidelity against
    ``real_fidelity_df`` (which MUST always be a TRAINING-partition raw
    dataframe, per Section 4's train-vs-synthetic fidelity protocol,
    regardless of what X_eval/y_eval are used for).

    ``config_overrides`` is passed straight into ``CDGANConfig(...)`` after
    the required structural fields (dataset_name, seed, encoded_dim,
    feature_group_slices) are filled in from the split -- so ANY
    CDGANConfig field can be swept by the caller (lambda_corr,
    lambda_feature, batch_size, n_critic, gp_every, corr_every, ...)
    without this module inventing its own parallel argument list. Do NOT
    include dataset_name/seed/encoded_dim/feature_group_slices in
    ``config_overrides`` -- they are set here from the split.
    """
    os.makedirs(out_dir, exist_ok=True)
    config_name = config_overrides.get("config_name", "full")

    config = CDGANConfig(
        dataset_name=dataset_name,
        seed=seed,
        encoded_dim=X_train.shape[1],
        feature_group_slices=fp.feature_group_slices,
        **{k: v for k, v in config_overrides.items() if k not in {"dataset_name", "seed", "encoded_dim", "feature_group_slices"}},
    )
    for k, v in ABLATION_FLAGS.get(config_name, {}).items():
        setattr(config, k, v)

    # Section 5/23: seed IMMEDIATELY before model construction, for every
    # (dataset, seed, configuration) triple -- including every grid point
    # of a sensitivity sweep and both arms of a regression check.
    set_all_seeds(seed)
    generator, discriminator = build_models(
        latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, cond_dim=config.cond_dim,
        encoded_dim=config.encoded_dim, feature_group_slices=fp.feature_group_slices,
        use_feature_head=config.use_feature_head,
    )
    pc = count_parameters(generator, discriminator)
    with open(os.path.join(out_dir, "param_counts.json"), "w") as f:
        json.dump(pc.as_dict(), f, indent=2)

    trainer = CDGANTrainer(config, generator, discriminator, fp.categorical_slices, out_dir)

    t0 = time.time()
    trainer.train(X_train, y_train)
    training_time_sec = time.time() - t0

    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    proportions = {int(c): float(cnt) / len(y_train) for c, cnt in zip(classes, counts)}

    t1 = time.time()
    _X_syn_raw, X_syn_proj, y_syn = trainer.generate_synthetic(len(y_train), proportions, seed=seed)
    synthetic_generation_time_sec = time.time() - t1

    X_train_clf, y_train_clf = build_tstr_training_set(
        X_syn_proj, y_syn, evaluation_mode=evaluation_mode, X_train_real=X_train, y_train_real=y_train,
    )
    utility_df = evaluate_tstr_per_classifier(
        X_train_clf, y_train_clf, X_eval, y_eval, seed=seed, evaluation_mode=evaluation_mode,
    )

    fake_df = inverse_transform(X_syn_proj, fp)
    # Rev 4: compare against the cleaned + train-imputed real frame, using the
    # columns the preprocessor actually kept (all-missing columns are dropped).
    fidelity = _aggregate_fidelity(
        impute_dataframe(real_fidelity_df, fp), fake_df, list(fp.numerical_cols), list(fp.categorical_cols),
        fp, X_train, X_syn_proj,
    )

    return ConfigurationResult(
        label=label,
        config=config,
        utility_df=utility_df,
        fidelity=fidelity,
        training_time_sec=training_time_sec,
        synthetic_generation_time_sec=synthetic_generation_time_sec,
    )
