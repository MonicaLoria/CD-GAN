"""
REV 4 NOTE: run_experiment.py no longer calls this module. --mode ablation
now trains each variant through run_experiment.run_single_seed (same
folders as --mode single, so the ablation "full" rows are the main-table
CD-GAN rows) and make_tables.py builds the absolute and delta tables.
Kept unchanged for reference to the Rev 3 behaviour.

ablation.py -- the four required ablation configurations (Section 14) and
their reporting tables (Section 15).

    Full CD-GAN : row ON, feature ON, GP ON (both heads),      corr ON
    w/o GP      : row ON, feature ON, GP OFF,                  corr ON
    w/o Corr    : row ON, feature ON, GP ON (both heads),      corr OFF
    w/o Dual    : row ON, feature OFF (single row critic only), GP ON (row only), corr ON

The exact flag combinations live in ``trainer.ABLATION_FLAGS`` so there is
one single source of truth (this module intentionally does not redefine
them, to avoid Section-14 vs Section-20 drift). "w/o Dual" uses
``DualHeadDiscriminator(use_feature_head=False)`` -- the feature head
module is never instantiated in that case, so its output cannot be
duplicated into the loss (the exact bug Section 14 asks to fix).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from model import build_models, count_parameters
from preprocessing import FittedPreprocessor
from trainer import ABLATION_FLAGS, CDGANConfig, CDGANTrainer, set_all_seeds
from evaluation import build_tstr_training_set, evaluate_tstr_per_classifier

ABLATION_CONFIGS = list(ABLATION_FLAGS.keys())  # ["full", "wo_gp", "wo_corr", "wo_dual"]


def run_single_ablation_variant(
    config_name: str,
    dataset_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fp: FittedPreprocessor,
    results_root: str,
    epochs: int,
    batch_size: int,
    lambda_gp: float,
    lambda_feature: float,
    lambda_corr: float,
    n_critic: int,
    gp_every: int,
    corr_every: int,
    use_class_weights: bool,
    evaluation_mode: str,
    device: str,
    resume: bool,
) -> pd.DataFrame:
    """Trains one ablation variant for one (dataset, seed) and returns its
    per-classifier absolute-performance rows (Section 15's first table).

    IMPORTANT (Section 5 of the follow-up revision): ``X_train``/``y_train``/
    ``X_test``/``y_test``/``fp`` must be the SAME raw-split-derived arrays
    for every config_name at a given seed -- the caller (``run_ablation_study``)
    is responsible for computing the split ONCE per seed and reusing it
    across all four variants, so ablations differ ONLY in the intended
    component, never in the underlying data split.
    """
    if config_name not in ABLATION_FLAGS:
        raise ValueError(f"Unknown ablation variant '{config_name}'. Known: {list(ABLATION_FLAGS)}")

    config = CDGANConfig(
        dataset_name=dataset_name,
        config_name=config_name,
        seed=seed,
        encoded_dim=X_train.shape[1],
        epochs=epochs,
        batch_size=batch_size,
        lambda_gp=lambda_gp,
        lambda_feature=lambda_feature,
        lambda_corr=lambda_corr,
        n_critic=n_critic,
        gp_every=gp_every,
        corr_every=corr_every,
        use_class_weights=use_class_weights,
        device=device,
        resume=resume,
        feature_group_slices=fp.feature_group_slices,
    )
    # apply_ablation_flags() inside CDGANTrainer will force use_gp/use_corr/
    # use_feature_head to the canonical combination for config_name.
    for k, v in ABLATION_FLAGS[config_name].items():
        setattr(config, k, v)

    # ---- Section 5: seed EVERY (dataset, seed, configuration) triple
    # immediately before model construction, so "full" and "wo_gp" (etc.)
    # at the SAME seed both start from the identical initial parameters
    # for the sub-networks they share (row head, generator), rather than
    # inheriting whatever RNG state training the previous variant left
    # behind. ----
    set_all_seeds(seed)
    generator, discriminator = build_models(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        cond_dim=config.cond_dim,
        encoded_dim=config.encoded_dim,
        feature_group_slices=fp.feature_group_slices,
        use_feature_head=config.use_feature_head,
    )

    out_dir = os.path.join(results_root, dataset_name, config_name, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    param_counts = count_parameters(generator, discriminator)
    with open(os.path.join(out_dir, "param_counts.json"), "w") as f:
        json.dump(param_counts.as_dict(), f, indent=2)

    trainer = CDGANTrainer(config, generator, discriminator, fp.categorical_slices, out_dir)

    t_train_start = time.time()
    trainer.train(X_train, y_train)
    training_time_sec = time.time() - t_train_start

    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    proportions = {int(c): float(cnt) / len(y_train) for c, cnt in zip(classes, counts)}

    t_gen_start = time.time()
    X_syn_raw, X_syn_proj, y_syn = trainer.generate_synthetic(
        n_samples=len(y_train), class_proportions=proportions, seed=seed
    )
    synthetic_generation_time_sec = time.time() - t_gen_start
    peak_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    with open(os.path.join(out_dir, "runtime.json"), "w") as f:
        json.dump({
            "training_time_sec": training_time_sec,
            "synthetic_generation_time_sec": synthetic_generation_time_sec,
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
        }, f, indent=2)

    X_train_clf, y_train_clf = build_tstr_training_set(
        X_syn_proj, y_syn, evaluation_mode=evaluation_mode, X_train_real=X_train, y_train_real=y_train
    )
    per_clf = evaluate_tstr_per_classifier(
        X_train_clf, y_train_clf, X_test, y_test, seed=seed, evaluation_mode=evaluation_mode
    )
    per_clf.insert(0, "configuration", config_name)
    per_clf.insert(1, "dataset", dataset_name)
    per_clf.insert(2, "seed", seed)

    per_clf.to_csv(os.path.join(out_dir, "utility.csv"), index=False)
    np.save(os.path.join(out_dir, "X_syn_projected_encoded.npy"), X_syn_proj)
    np.save(os.path.join(out_dir, "X_syn_raw_encoded.npy"), X_syn_raw)
    np.save(os.path.join(out_dir, "y_syn.npy"), y_syn)

    return per_clf


def run_ablation_study(
    dataset_name: str,
    seeds: List[int],
    X_train_by_seed: Dict[int, np.ndarray],
    y_train_by_seed: Dict[int, np.ndarray],
    X_test_by_seed: Dict[int, np.ndarray],
    y_test_by_seed: Dict[int, np.ndarray],
    fp_by_seed: Dict[int, FittedPreprocessor],
    results_root: str,
    epochs: int,
    batch_size: int,
    lambda_gp: float,
    lambda_feature: float,
    lambda_corr: float,
    n_critic: int,
    gp_every: int,
    corr_every: int,
    use_class_weights: bool,
    evaluation_mode: str,
    device: str,
    resume: bool,
    variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Runs all four (or a subset of) ablation variants across all seeds.

    ``X_train_by_seed``/``y_train_by_seed``/``X_test_by_seed``/
    ``y_test_by_seed``/``fp_by_seed`` must each hold ONE split per seed,
    computed ONCE by the caller and reused across every variant at that
    seed (Section 5/8) -- this function never re-splits."""
    variants = variants or ABLATION_CONFIGS
    all_rows = []
    for config_name in variants:
        for seed in seeds:
            rows = run_single_ablation_variant(
                config_name=config_name,
                dataset_name=dataset_name,
                seed=seed,
                X_train=X_train_by_seed[seed],
                y_train=y_train_by_seed[seed],
                X_test=X_test_by_seed[seed],
                y_test=y_test_by_seed[seed],
                fp=fp_by_seed[seed],
                results_root=results_root,
                epochs=epochs,
                batch_size=batch_size,
                lambda_gp=lambda_gp,
                lambda_feature=lambda_feature,
                lambda_corr=lambda_corr,
                n_critic=n_critic,
                gp_every=gp_every,
                corr_every=corr_every,
                use_class_weights=use_class_weights,
                evaluation_mode=evaluation_mode,
                device=device,
                resume=resume,
            )
            all_rows.append(rows)
    absolute_table = pd.concat(all_rows, ignore_index=True)
    absolute_table.to_csv(os.path.join(results_root, dataset_name, "ablation_absolute_table.csv"), index=False)
    return absolute_table


# ======================================================================
# Section 15: reporting -- absolute table, aggregated table, delta table
# (NEVER mixed in the same column/table)
# ======================================================================
def build_absolute_table(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Configuration | Dataset | Seed | Classifier | F1 | ROC-AUC | PR-AUC"""
    cols = ["configuration", "dataset", "seed", "classifier", "f1", "roc_auc", "pr_auc"]
    return per_run_df[cols].copy()


def build_aggregated_table(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Configuration | Classifier | F1 mean+-SD | ROC mean+-SD | PR mean+-SD
    (aggregated ACROSS SEEDS, per dataset)."""
    grouped = per_run_df.groupby(["dataset", "configuration", "classifier"])[
        ["f1", "roc_auc", "pr_auc"]
    ].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def build_delta_table(per_run_df: pd.DataFrame, baseline_config: str = "full") -> pd.DataFrame:
    """SEPARATE table (Section 15): delta-F1/ROC/PR vs the baseline
    configuration, per dataset/classifier, aggregated across seeds. Never
    mixed with absolute scores in the same column."""
    agg = per_run_df.groupby(["dataset", "configuration", "classifier"])[
        ["f1", "roc_auc", "pr_auc"]
    ].mean().reset_index()

    baseline = agg[agg["configuration"] == baseline_config].drop(columns=["configuration"])
    baseline = baseline.rename(columns={"f1": "f1_base", "roc_auc": "roc_auc_base", "pr_auc": "pr_auc_base"})

    merged = agg.merge(baseline, on=["dataset", "classifier"], how="left")
    merged["delta_f1_vs_full"] = merged["f1"] - merged["f1_base"]
    merged["delta_roc_auc_vs_full"] = merged["roc_auc"] - merged["roc_auc_base"]
    merged["delta_pr_auc_vs_full"] = merged["pr_auc"] - merged["pr_auc_base"]

    return merged[
        ["dataset", "configuration", "classifier",
         "delta_f1_vs_full", "delta_roc_auc_vs_full", "delta_pr_auc_vs_full"]
    ]
