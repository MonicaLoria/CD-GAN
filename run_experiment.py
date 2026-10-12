"""
run_experiment.py -- single CLI entry point for every experiment type.

    python run_experiment.py --mode sanity
    python run_experiment.py --mode single   --dataset SICK --data-path ... --seeds 0 --epochs 8
    python run_experiment.py --mode single   --dataset SICK --data-path ... --seeds 0 1 2 --epochs 1000
    python run_experiment.py --mode ablation --dataset SICK --data-path ... --seeds 0 1 2 --epochs 1000

See README.md for the full set of example commands (kept in exact sync
with the --help output of this script -- every flag documented there is
defined below, nothing is invented).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch

from datasets import load_dataset
from evaluation import build_tstr_training_set
from quality_metrics import evaluate_synthetic
from run_utils import export_split, package_versions, split_fingerprint, write_json
from model import build_models, count_parameters
from preprocessing import (
    impute_dataframe,
    inverse_transform,
    prepare_seed_split,
    preprocessing_report,
    save_config_snapshot,
)
from regression_check import run_regression_check
from sanity_checks import run_all_sanity_checks
from sensitivity import (
    DEFAULT_BATCH_SIZE_GRID,
    DEFAULT_CORR_EVERY_GRID,
    DEFAULT_GP_EVERY_GRID,
    DEFAULT_LAMBDA_GP_GRID,
    DEFAULT_LAMBDA_CORR_GRID,
    DEFAULT_LAMBDA_FEATURE_GRID,
    run_sensitivity_stage,
    run_staged_sensitivity,
)
from trainer import ABLATION_FLAGS, CDGANConfig, CDGANTrainer, set_all_seeds


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CD-GAN reviewer-revision experiment runner.")
    p.add_argument("--mode", type=str, default="single",
                    choices=["sanity", "single", "ablation", "sensitivity", "regression_check"])

    p.add_argument("--dataset", type=str, default=None, help="SICK | NOMAO | MV | KDDCUP09_UPSELLING | SPEC (with --data-path spec.json)")
    p.add_argument("--data-path", type=str, default=None, help="path to the dataset's raw file (.arff/.csv)")
    p.add_argument("--config-name", type=str, default="full", choices=list(ABLATION_FLAGS.keys()),
                    help="ablation variant to train in --mode single (default: full CD-GAN)")
    p.add_argument("--ablation-variants", type=str, nargs="+", default=list(ABLATION_FLAGS.keys()),
                    choices=list(ABLATION_FLAGS.keys()), help="which variants to run in --mode ablation")

    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--validation-size", type=float, default=0.2,
                    help="--mode sensitivity/regression_check only: fraction of the DEV (1-test_size) "
                         "partition set aside as an internal VALIDATION partition (Section 15) -- the "
                         "final TEST partition is never touched by either mode")

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128,
                    help="NOTE: manuscript reports 64, prior code used 128 -- set explicitly, not silently resolved")
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--cond-dim", type=int, default=1)
    p.add_argument("--learning-rate-g", type=float, default=2e-4)
    p.add_argument("--learning-rate-d", type=float, default=2e-4)
    p.add_argument("--adam-beta1", type=float, default=0.5)
    p.add_argument("--adam-beta2", type=float, default=0.9)
    p.add_argument("--n-critic", type=int, default=2)

    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--lambda-feature", type=float, default=1.0)
    p.add_argument("--lambda-corr", type=float, default=0.05,
                    help="NOTE: manuscript reports 1.0, prior code used 0.05 -- set explicitly, not silently resolved")
    p.add_argument("--gp-every", type=int, default=5,
                    help="compute GP every N critic steps (production default restored to 5, matching your "
                         "previous implementation -- was silently left at 1 in the prior pass of this package)")
    p.add_argument("--corr-every", type=int, default=5,
                    help="compute L_corr every N generator steps (production default restored to 5, "
                         "matching your previous implementation)")

    p.add_argument("--dependency-loss", type=str, default="covariance", choices=["covariance", "pearson"],
                    help="Rev 4: dependency regularizer used when use_corr=True (ablation variant "
                         "'pearson_corr' sets 'pearson' automatically)")
    p.add_argument("--synthetic-class-ratio", type=str, default="train", choices=["train", "balanced"],
                    help="Rev 4: class proportions of the generated data ('train' keeps the real training ratio)")
    p.add_argument("--skip-existing", action="store_true", default=False,
                    help="Rev 4: reuse a (config, seed) run folder that already has quality_summary.csv "
                         "(always on in --mode ablation)")
    p.add_argument("--export-splits", type=str, default=None,
                    help="Rev 4: also write each seed's raw train/test CSVs here, for external baselines")
    p.add_argument("--use-class-weights", action="store_true", default=False,
                    help="OFF by default -- the manuscript's mathematical formulation is unweighted")
    p.add_argument("--measure-grad-cosine", action="store_true", default=False,
                    help="Section 22: OFF by default. OPTIONAL diagnostic-only cosine-similarity "
                         "measurement between the generator gradients from the row-adversarial term "
                         "alone vs the feature-adversarial term alone. Purely observational -- gradients "
                         "are NEVER modified (no PCGrad, no gradient surgery).")
    p.add_argument("--grad-cosine-every", type=int, default=50,
                    help="evaluate the Section 22 diagnostic every N global_generator_step values "
                         "(only relevant when --measure-grad-cosine is set)")

    p.add_argument("--evaluation", type=str, default="TSTR", choices=["TSTR", "augmented"])
    p.add_argument("--n-synthetic", type=int, default=None, help="default: N_synthetic = N_train")

    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume", action="store_true", default=False)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--results-root", type=str, default="results")

    # ---- Section 14: --mode regression_check ----
    p.add_argument("--regression-tolerance", type=float, default=0.05,
                    help="relative-degradation threshold (default 5%%) beyond which a metric is flagged "
                         "as regressed in --mode regression_check")
    for _axis in ["lambda-corr", "lambda-feature", "n-critic", "gp-every", "corr-every"]:
        dest_axis = _axis.replace("-", "_")
        val_type = int if _axis in {"n-critic", "gp-every", "corr-every"} else float
        p.add_argument(f"--reference-{_axis}", type=val_type, default=None,
                        help=f"--mode regression_check: override {dest_axis} for the REFERENCE configuration "
                             f"(default: same as --{_axis})")
        p.add_argument(f"--final-{_axis}", type=val_type, default=None,
                        help=f"--mode regression_check: override {dest_axis} for the FINAL configuration "
                             f"(default: same as --{_axis})")
    p.add_argument("--reference-batch-size", type=int, default=None,
                    help="--mode regression_check: override batch_size for the REFERENCE configuration")
    p.add_argument("--final-batch-size", type=int, default=None,
                    help="--mode regression_check: override batch_size for the FINAL configuration")

    # ---- Sections 15-20: --mode sensitivity ----
    p.add_argument("--sensitivity-stage", type=str, default="staged",
                    choices=["staged", "lambda_corr", "lambda_feature", "batch_size",
                             "gp_every", "corr_every", "lambda_gp"],
                    help="'staged' runs Stage A -> Stage B -> (optional) Stage C in sequence (Section 16); "
                         "otherwise sweeps exactly one named axis")
    p.add_argument("--lambda-corr-grid", type=float, nargs="+", default=None,
                    help=f"default: {DEFAULT_LAMBDA_CORR_GRID}")
    p.add_argument("--lambda-feature-grid", type=float, nargs="+", default=None,
                    help=f"default: {DEFAULT_LAMBDA_FEATURE_GRID}")
    p.add_argument("--batch-size-grid", type=int, nargs="+", default=None,
                    help=f"default: {DEFAULT_BATCH_SIZE_GRID}")
    p.add_argument("--gp-every-grid", type=int, nargs="+", default=None,
                    help=f"Rev 4, default: {DEFAULT_GP_EVERY_GRID}")
    p.add_argument("--corr-every-grid", type=int, nargs="+", default=None,
                    help=f"Rev 4, default: {DEFAULT_CORR_EVERY_GRID}")
    p.add_argument("--lambda-gp-grid", type=float, nargs="+", default=None,
                    help=f"Rev 4, default: {DEFAULT_LAMBDA_GP_GRID}")
    p.add_argument("--run-stage-c", action="store_true", default=False,
                    help="--mode sensitivity --sensitivity-stage staged: also run Stage C (batch_size "
                         "comparison) after Stages A/B")
    return p


def print_environment_banner() -> None:
    print("=" * 70)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Python version  : {platform.python_version()}")
    print(f"Platform        : {platform.platform()}")
    print("=" * 70)


def save_package_versions(out_path: str) -> None:
    import sklearn
    import scipy
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(versions, f, indent=2)


# ======================================================================
def _runtime_from_history(out_dir: str) -> Dict[str, float]:
    """Rev 4 (reviewer point 13): per-epoch wall-clock and peak GPU memory
    come from training_history.csv, which resets the CUDA peak counter at
    the start of every epoch (so values never leak between runs)."""
    path = os.path.join(out_dir, "training_history.csv")
    if not os.path.exists(path):
        return {"seconds_per_epoch": float("nan"), "peak_cuda_memory_mb_training": float("nan")}
    h = pd.read_csv(path)
    return {
        "seconds_per_epoch": float(h["epoch_time_sec"].mean()) if "epoch_time_sec" in h else float("nan"),
        "peak_cuda_memory_mb_training": float(h["peak_cuda_memory_mb"].max()) if "peak_cuda_memory_mb" in h else float("nan"),
        "epochs_completed": int(len(h)),
    }


def synthetic_class_proportions(y_train: np.ndarray, mode: str) -> Dict[int, float]:
    """Rev 4 (reviewer point 9). 'train' preserves the real training class
    ratio (default, used for all TSTR tables); 'balanced' requests 50/50.
    The condition c is the scalar binary label (cond_dim = 1), so only
    binary targets are supported."""
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f"CD-GAN conditioning supports binary targets only; found classes {classes.tolist()}")
    if mode == "balanced":
        return {0: 0.5, 1: 0.5}
    return {int(c): float(n) / len(y_train) for c, n in zip(classes, counts)}


def run_single_seed(
    args: argparse.Namespace,
    dataset_name: str,
    seed: int,
    df,
    target_column: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_mapping: Optional[Dict],
    config_name: Optional[str] = None,
) -> Dict:
    """Rev 4: trains ONE configuration (full CD-GAN or an ablation variant)
    for one seed and scores it with quality_metrics.evaluate_synthetic --
    the same evaluator run_baselines.py uses. Both --mode single and
    --mode ablation go through this function, and both write to
    results/<DATASET>/<config_name>/seed<k>/, so the ablation 'full' rows
    ARE the main-table CD-GAN rows (reviewer point 3)."""
    config_name = config_name or args.config_name
    out_dir = os.path.join(args.results_root, dataset_name, config_name, f"seed{seed}")
    done_flag = os.path.join(out_dir, "quality_summary.csv")
    if getattr(args, "skip_existing", False) and os.path.exists(done_flag):
        print(f"[{dataset_name} {config_name} seed={seed}] already complete -> reusing {out_dir}")
        return {"out_dir": out_dir, "summary": pd.read_csv(done_flag).iloc[0].to_dict()}

    set_all_seeds(seed)
    split = prepare_seed_split(
        df, target_column, categorical_cols, numerical_cols, dataset_name,
        seed=seed, test_size=args.test_size, target_mapping=target_mapping,
    )
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    fp = split["preprocessor"]

    print(
        f"[{dataset_name} seed={seed}] encoded_dim={fp.encoded_dim}, "
        f"n_original_feature_groups={len(fp.original_feature_groups)}, "
        f"n_categorical_cols={len(fp.categorical_cols)}, n_numerical_cols={len(fp.numerical_cols)}"
    )
    classes_tr, counts_tr = np.unique(y_train.astype(int), return_counts=True)
    total_tr = counts_tr.sum()
    print(
        f"[{dataset_name} seed={seed}] TRAIN class distribution: "
        + ", ".join(f"{int(c)}:{int(n)} ({100.0 * n / total_tr:.2f}%)" for c, n in zip(classes_tr, counts_tr))
    )

    for sub in ["checkpoints", "configs"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    fp.save(os.path.join(out_dir, "configs", "preprocessor.joblib"))
    save_config_snapshot(fp, os.path.join(out_dir, "configs", "preprocessor_summary.json"))
    prep = preprocessing_report(fp, split["train_df"], split["test_df"], args.test_size)
    write_json(prep, os.path.join(out_dir, "configs", "preprocessing_report.json"))
    pd.DataFrame(prep["columns"]).to_csv(os.path.join(out_dir, "configs", "preprocessing_columns.csv"), index=False)
    write_json(split_fingerprint(split["train_df"], split["test_df"], split["train_row_ids"], split["test_row_ids"]), os.path.join(out_dir, "split_fingerprint.json"))
    write_json(package_versions(), os.path.join(out_dir, "package_versions.json"))
    if getattr(args, "export_splits", None):
        export_split(split, os.path.join(args.export_splits, dataset_name, f"seed{seed}"), target_column)

    config = CDGANConfig(
        dataset_name=dataset_name,
        config_name=config_name,
        seed=seed,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        learning_rate_g=args.learning_rate_g,
        learning_rate_d=args.learning_rate_d,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        lambda_gp=args.lambda_gp,
        lambda_feature=args.lambda_feature,
        lambda_corr=args.lambda_corr,
        n_critic=args.n_critic,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gp_every=args.gp_every,
        corr_every=args.corr_every,
        dependency_loss=args.dependency_loss,
        use_class_weights=args.use_class_weights,
        measure_grad_cosine=args.measure_grad_cosine,
        grad_cosine_every=args.grad_cosine_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        device=args.device,
        encoded_dim=X_train.shape[1],
        feature_group_slices=fp.feature_group_slices,
    )
    for k, v in ABLATION_FLAGS.get(config_name, {}).items():
        setattr(config, k, v)

    set_all_seeds(seed)
    generator, discriminator = build_models(
        latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, cond_dim=config.cond_dim,
        encoded_dim=config.encoded_dim, feature_group_slices=fp.feature_group_slices,
        use_feature_head=config.use_feature_head,
    )
    pc = count_parameters(generator, discriminator)
    write_json(pc.as_dict(), os.path.join(out_dir, "param_counts.json"))

    trainer = CDGANTrainer(config, generator, discriminator, fp.categorical_slices, out_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(trainer.device)
    t_train_start = time.time()
    trainer.train(X_train, y_train)
    training_time_sec = time.time() - t_train_start

    n_synthetic = args.n_synthetic or len(y_train)
    proportions = synthetic_class_proportions(y_train, args.synthetic_class_ratio)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(trainer.device)
    t_gen_start = time.time()
    X_syn_raw, X_syn_proj, y_syn = trainer.generate_synthetic(n_synthetic, proportions, seed=seed)
    synthetic_generation_time_sec = time.time() - t_gen_start
    peak_gen_mb = (torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 2)
                   if torch.cuda.is_available() else float("nan"))

    np.save(os.path.join(out_dir, "X_syn_raw_encoded.npy"), X_syn_raw)
    np.save(os.path.join(out_dir, "X_syn_projected_encoded.npy"), X_syn_proj)
    np.save(os.path.join(out_dir, "y_syn.npy"), y_syn)
    fake_df = inverse_transform(X_syn_proj, fp)
    fake_df.assign(**{target_column: y_syn.astype(int)}).to_csv(
        os.path.join(out_dir, "synthetic_decoded.csv"), index=False)

    hist_rt = _runtime_from_history(out_dir)
    runtime = {
        "training_time_sec": training_time_sec,
        "epochs": config.epochs,
        **hist_rt,
        "synthetic_generation_time_sec": synthetic_generation_time_sec,
        "n_synthetic_generated": int(n_synthetic),
        "peak_cuda_memory_mb_generation": peak_gen_mb,
        "peak_cuda_memory_mb": float(np.nanmax([hist_rt.get("peak_cuda_memory_mb_training", np.nan), peak_gen_mb]))
        if torch.cuda.is_available() else float("nan"),
        "total_parameters": pc.as_dict()["total_parameters"],
    }
    write_json(runtime, os.path.join(out_dir, "runtime.json"))

    counts_syn = {int(c): int((y_syn.astype(int) == c).sum()) for c in (0, 1)}
    write_json({
        "synthetic_class_ratio_mode": args.synthetic_class_ratio,
        "requested_proportions": proportions,
        "generated_counts": counts_syn,
        "n_synthetic": int(n_synthetic),
        "condition": "scalar binary label c in {0,1}, cond_dim=1; z ~ N(0, I_latent_dim) drawn fresh per row",
        "categorical_output": "argmax per one-hot block, then inverse one-hot / inverse z-score",
    }, os.path.join(out_dir, "generation_report.json"))
    write_json({**config.as_dict(), "synthetic_class_ratio": args.synthetic_class_ratio,
                "n_synthetic": int(n_synthetic), "test_size": args.test_size,
                "evaluation": args.evaluation}, os.path.join(out_dir, "run_config.json"))

    # ---- Rev 4: ONE evaluator (quality_metrics) for CD-GAN and baselines ----
    real_train_imputed = impute_dataframe(split["train_df"], fp)
    X_clf, y_clf = build_tstr_training_set(
        X_syn_proj, y_syn, evaluation_mode=args.evaluation, X_train_real=X_train, y_train_real=y_train,
    )
    summary = evaluate_synthetic(
        method="CD-GAN" if config_name == "full" else f"CD-GAN[{config_name}]",
        dataset=dataset_name, seed=seed, out_dir=out_dir, fp=fp,
        real_train_df=real_train_imputed, y_train_real=y_train, X_train_real=X_train,
        X_test_real=X_test, y_test_real=y_test,
        fake_df=fake_df, X_syn=X_syn_proj, y_syn=y_syn,
        utility_protocol=args.evaluation, X_clf_train=X_clf, y_clf_train=y_clf,
    )
    print(f"[{dataset_name} {config_name} seed={seed}] F1={summary['util_f1']:.4f} "
          f"PR-AUC={summary['util_pr_auc']:.4f} WDnum={summary['num_wd_norm_mean']:.4f} "
          f"TVcat={summary['cat_tv_distance_mean']:.4f} assocMAE={summary['assoc_mae']:.4f}")
    return {"out_dir": out_dir, "summary": summary}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    print_environment_banner()

    if args.mode == "sanity":
        run_all_sanity_checks(device=args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        return

    if args.dataset is None or args.data_path is None:
        parser.error("--dataset and --data-path are required for --mode single/ablation")

    spec = load_dataset(args.dataset, args.data_path)
    print(f"Loaded dataset '{spec.dataset_name}': {spec.dataframe.shape[0]} rows, "
          f"{len(spec.categorical_cols)} categorical cols, {len(spec.numerical_cols)} numerical cols")

    os.makedirs(args.results_root, exist_ok=True)

    if args.mode in {"single", "ablation"}:
        # Rev 4: both modes use run_single_seed and the same folder layout.
        # Ablation variants reuse any already-complete run (including the
        # 'full' run from --mode single), so Table 4 (main) and Table 5
        # (ablation) are built from the SAME files by make_tables.py.
        variants = [args.config_name] if args.mode == "single" else args.ablation_variants
        if args.mode == "ablation":
            args.skip_existing = True
        rows = []
        for variant in variants:
            for seed in args.seeds:
                res = run_single_seed(
                    args, spec.dataset_name, seed, spec.dataframe, spec.target_column,
                    spec.categorical_cols, spec.numerical_cols, spec.target_mapping, config_name=variant,
                )
                rows.append({"configuration": variant, **res["summary"]})
        summary = pd.DataFrame(rows)
        key_cols = [c for c in ["util_f1", "util_roc_auc", "util_pr_auc", "util_mcc", "util_minority_recall",
                                "num_wd_norm_mean", "cat_tv_distance_mean", "assoc_mae",
                                "detection_auc_gbm", "syn_duplicate_ratio", "dcr_mean"] if c in summary.columns]
        agg = summary.groupby("configuration")[key_cols].agg(["mean", "std"])
        agg.columns = [f"{m}_{st}" for m, st in agg.columns]
        out_csv = os.path.join(args.results_root, spec.dataset_name, f"{args.mode}_summary_mean_sd.csv")
        agg.reset_index().to_csv(out_csv, index=False)
        print(agg.T.to_string())
        print(f"\nWrote {out_csv}. Build paper tables with: python make_tables.py --results-root "
              f"{args.results_root} --out-dir paper_tables")

    elif args.mode == "regression_check":
        # ---- Section 14: performance regression guard. Both arms train
        # on the IDENTICAL internal GAN-TRAIN/VALIDATION split (fit once
        # per seed); the final TEST partition is never touched here. ----
        base_overrides: Dict = dict(
            latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
            learning_rate_g=args.learning_rate_g, learning_rate_d=args.learning_rate_d,
            adam_betas=(args.adam_beta1, args.adam_beta2),
            lambda_gp=args.lambda_gp, lambda_feature=args.lambda_feature, lambda_corr=args.lambda_corr,
            n_critic=args.n_critic, batch_size=args.batch_size, epochs=args.epochs,
            gp_every=args.gp_every, corr_every=args.corr_every, use_class_weights=args.use_class_weights,
            measure_grad_cosine=args.measure_grad_cosine, grad_cosine_every=args.grad_cosine_every,
            checkpoint_every=args.checkpoint_every, resume=False, device=args.device,
            config_name=args.config_name, dependency_loss=args.dependency_loss,
        )
        axis_cli_names = {
            "lambda_corr": ("reference_lambda_corr", "final_lambda_corr"),
            "lambda_feature": ("reference_lambda_feature", "final_lambda_feature"),
            "n_critic": ("reference_n_critic", "final_n_critic"),
            "gp_every": ("reference_gp_every", "final_gp_every"),
            "corr_every": ("reference_corr_every", "final_corr_every"),
            "batch_size": ("reference_batch_size", "final_batch_size"),
        }
        reference_overrides: Dict = {}
        final_overrides: Dict = {}
        for field_name, (ref_attr, final_attr) in axis_cli_names.items():
            ref_val = getattr(args, ref_attr)
            final_val = getattr(args, final_attr)
            if ref_val is not None:
                reference_overrides[field_name] = ref_val
            if final_val is not None:
                final_overrides[field_name] = final_val

        for seed in args.seeds:
            run_regression_check(
                dataset_name=spec.dataset_name, df=spec.dataframe, target_column=spec.target_column,
                categorical_cols=spec.categorical_cols, numerical_cols=spec.numerical_cols,
                target_mapping=spec.target_mapping, seed=seed, test_size=args.test_size,
                validation_size=args.validation_size, base_overrides=base_overrides,
                reference_overrides=reference_overrides, final_overrides=final_overrides,
                results_root=args.results_root, tolerance=args.regression_tolerance,
            )

    elif args.mode == "sensitivity":
        # ---- Sections 15-20: leak-free sensitivity analysis. Every run
        # trains on the internal GAN-TRAIN partition and is scored on the
        # internal VALIDATION partition; the final TEST partition is never
        # touched. ----
        base_overrides = dict(
            latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
            learning_rate_g=args.learning_rate_g, learning_rate_d=args.learning_rate_d,
            adam_betas=(args.adam_beta1, args.adam_beta2),
            lambda_gp=args.lambda_gp, lambda_feature=args.lambda_feature, lambda_corr=args.lambda_corr,
            n_critic=args.n_critic, batch_size=args.batch_size, epochs=args.epochs,
            gp_every=args.gp_every, corr_every=args.corr_every, use_class_weights=args.use_class_weights,
            measure_grad_cosine=args.measure_grad_cosine, grad_cosine_every=args.grad_cosine_every,
            checkpoint_every=args.checkpoint_every, resume=False, device=args.device,
            config_name=args.config_name, dependency_loss=args.dependency_loss,
        )
        for seed in args.seeds:
            if args.sensitivity_stage == "staged":
                run_staged_sensitivity(
                    dataset_name=spec.dataset_name, df=spec.dataframe, target_column=spec.target_column,
                    categorical_cols=spec.categorical_cols, numerical_cols=spec.numerical_cols,
                    target_mapping=spec.target_mapping, seed=seed, test_size=args.test_size,
                    validation_size=args.validation_size, base_overrides=base_overrides,
                    results_root=args.results_root,
                    lambda_corr_grid=args.lambda_corr_grid, lambda_feature_grid=args.lambda_feature_grid,
                    batch_size_grid=args.batch_size_grid, run_stage_c=args.run_stage_c,
                )
            else:
                axis_name = args.sensitivity_stage
                axis_values = {
                    "lambda_corr": args.lambda_corr_grid or DEFAULT_LAMBDA_CORR_GRID,
                    "lambda_feature": args.lambda_feature_grid or DEFAULT_LAMBDA_FEATURE_GRID,
                    "batch_size": args.batch_size_grid or DEFAULT_BATCH_SIZE_GRID,
                    "gp_every": args.gp_every_grid or DEFAULT_GP_EVERY_GRID,
                    "corr_every": args.corr_every_grid or DEFAULT_CORR_EVERY_GRID,
                    "lambda_gp": args.lambda_gp_grid or DEFAULT_LAMBDA_GP_GRID,
                }[axis_name]
                run_sensitivity_stage(
                    dataset_name=spec.dataset_name, df=spec.dataframe, target_column=spec.target_column,
                    categorical_cols=spec.categorical_cols, numerical_cols=spec.numerical_cols,
                    target_mapping=spec.target_mapping, seed=seed, test_size=args.test_size,
                    validation_size=args.validation_size, base_overrides=base_overrides,
                    axis_name=axis_name, axis_values=axis_values, results_root=args.results_root,
                )


if __name__ == "__main__":
    main()
