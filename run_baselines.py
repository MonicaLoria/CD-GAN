"""
run_baselines.py -- Rev 4: baselines on EXACTLY the same split, preprocessing
and evaluator as CD-GAN (reviewer points 8 and 18; baseline provenance).

For each (dataset, seed) this script:
  1. builds the split with preprocessing.prepare_seed_split -- the same
     function, seed and test_size as run_experiment.py, so the real train
     and test rows are identical (split_fingerprint.json proves it);
  2. fits the baseline on the real TRAIN partition only;
  3. generates a fully synthetic table with the same size and class ratio
     as the real training partition (unless --synthetic-class-ratio balanced);
  4. scores it with quality_metrics.evaluate_synthetic -- the evaluator
     CD-GAN uses -- and writes results/<DATASET>/baseline_<METHOD>/seed<k>/;
  5. records library versions, every hyperparameter, epochs, sample counts
     and wall-clock time in run_config.json / runtime.json.

Methods
-------
REAL            reference only (TRTR): classifiers trained on the real
                training partition. Not a competitor; excluded from the
                statistical tests by make_tables.py.
SMOTE, ADASYN   imbalanced-learn, in the train-fitted encoded space. Each
                class's target count is doubled and ONLY the newly
                interpolated rows are kept, giving a fully synthetic table
                (TSTR, like every other method). Interpolated one-hot blocks
                are projected to valid categories by argmax.
GaussianCopula, CTGAN, TVAE, CopulaGAN
                SDV single-table synthesizers fit on the raw training frame
                (with the target as a categorical column) and sampled
                conditionally on the class label to hit the exact class counts.
EXTERNAL:<NAME> any generator run outside this package (TWAE, TabDDPM,
                CTAB-GAN+, TABULA, ...). Train it on
                <export_dir>/<DATASET>/seed<k>/train.csv (written by
                run_experiment.py --export-splits or by this script with
                --export-splits) and save its synthetic table, same columns
                incl. the raw target, as
                <external_dir>/<DATASET>/seed<k>/<NAME>.csv.
                It is then scored by the identical evaluator.

Examples
--------
    python run_baselines.py --dataset SICK --data-path dataset_38_sick.arff \
        --methods REAL SMOTE ADASYN CTGAN TVAE GaussianCopula --seeds 0 1 2 3 4 \
        --sdv-epochs 300 --results-root results --export-splits splits

    python run_baselines.py --dataset SICK --data-path dataset_38_sick.arff \
        --methods EXTERNAL:TWAE --external-dir external_synth --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from datasets import load_dataset
from preprocessing import (
    impute_dataframe,
    inverse_transform,
    prepare_seed_split,
    project_categorical_blocks,
    transform_with_preprocessor,
)
from quality_metrics import evaluate_synthetic
from run_utils import export_split, package_versions, split_fingerprint, write_json

SDV_METHODS = {"GAUSSIANCOPULA", "CTGAN", "TVAE", "COPULAGAN"}
CANONICAL = {"GAUSSIANCOPULA": "GaussianCopula", "CTGAN": "CTGAN", "TVAE": "TVAE",
             "COPULAGAN": "CopulaGAN", "SMOTE": "SMOTE", "ADASYN": "ADASYN", "REAL": "REAL"}


def target_counts(y: np.ndarray, n_total: int, mode: str) -> Dict[int, int]:
    y = np.asarray(y).astype(int)
    if mode == "balanced":
        half = n_total // 2
        return {0: n_total - half, 1: half}
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): cnt / len(y) * n_total for c, cnt in zip(classes, counts)}
    out = {c: int(round(v)) for c, v in raw.items()}
    out[max(out)] += n_total - sum(out.values())
    return out


# ----------------------------------------------------------------------
def oversampler_synthetic(method: str, X: np.ndarray, y: np.ndarray, counts: Dict[int, int],
                          categorical_slices, seed: int, k_neighbors: int = 5
                          ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    from imblearn.over_sampling import ADASYN, SMOTE

    y = np.asarray(y).astype(int)
    labels, have = np.unique(y, return_counts=True)
    original = {int(c): int(n) for c, n in zip(labels, have)}
    # request (original + needed) rows per class so enough NEW rows exist
    strategy = {c: original[c] + max(counts[c], 1) for c in original}
    k = max(1, min(k_neighbors, int(have.min()) - 1))
    if method == "SMOTE":
        sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    else:
        sampler = ADASYN(sampling_strategy=strategy, n_neighbors=k, random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    X_new_parts, y_new_parts = [], []
    rng = np.random.default_rng(seed)
    for c in original:
        # imbalanced-learn returns the original rows first, then new rows
        X_new = X_res[len(X):][y_res[len(X):] == c]
        if len(X_new) < counts[c]:
            warnings.warn(f"{method}: class {c} produced {len(X_new)} new rows, needed {counts[c]}; "
                          "sampling with replacement from the new rows")
            if len(X_new) == 0:
                raise RuntimeError(f"{method} produced no rows for class {c}")
            X_new = X_new[rng.integers(0, len(X_new), counts[c])]
        else:
            X_new = X_new[rng.choice(len(X_new), counts[c], replace=False)]
        X_new_parts.append(X_new)
        y_new_parts.append(np.full(counts[c], c))
    X_syn = project_categorical_blocks(np.vstack(X_new_parts), categorical_slices)
    y_syn = np.concatenate(y_new_parts).astype(np.float32)
    perm = rng.permutation(len(y_syn))
    params = {"library": "imbalanced-learn", "space": "train-fitted encoded space (one-hot | z-score)",
              "k_neighbors": k, "sampling": "new interpolated rows only (fully synthetic)",
              "categorical_postprocess": "argmax per one-hot block"}
    return X_syn[perm], y_syn[perm], params


def sdv_synthetic(method: str, train_df: pd.DataFrame, target_column: str, counts_raw: Dict[str, int],
                  seed: int, epochs: int, batch_size: int) -> Tuple[pd.DataFrame, Dict]:
    import torch
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition
    from sdv.single_table import (CopulaGANSynthesizer, CTGANSynthesizer,
                                  GaussianCopulaSynthesizer, TVAESynthesizer)

    np.random.seed(seed)
    torch.manual_seed(seed)
    cls = {"GAUSSIANCOPULA": GaussianCopulaSynthesizer, "CTGAN": CTGANSynthesizer,
           "TVAE": TVAESynthesizer, "COPULAGAN": CopulaGANSynthesizer}[method]
    df = train_df.copy()
    df[target_column] = df[target_column].astype(str)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(target_column, sdtype="categorical")
    kwargs = {}
    if method in {"CTGAN", "TVAE", "COPULAGAN"}:
        kwargs = {"epochs": epochs, "batch_size": batch_size, "cuda": torch.cuda.is_available()}
    synth = cls(metadata, **kwargs)
    synth.fit(df)
    conditions = [Condition(num_rows=n, column_values={target_column: lab}) for lab, n in counts_raw.items() if n > 0]
    fake = synth.sample_from_conditions(conditions)
    params = {"library": "sdv", "synthesizer": cls.__name__, **{k: v for k, v in kwargs.items()},
              "other_hyperparameters": "SDV defaults", "sampling": "conditional on class label"}
    return fake.sample(frac=1.0, random_state=seed).reset_index(drop=True), params


# ----------------------------------------------------------------------
def run_one(args, spec, method_arg: str, seed: int) -> Dict:
    ext_name = method_arg.split(":", 1)[1] if method_arg.upper().startswith("EXTERNAL:") else None
    key = "EXTERNAL" if ext_name else method_arg.upper()
    method = ext_name or CANONICAL.get(key, method_arg)
    out_dir = os.path.join(args.results_root, spec.dataset_name, f"baseline_{method}", f"seed{seed}")
    if args.skip_existing and os.path.exists(os.path.join(out_dir, "quality_summary.csv")):
        print(f"[{spec.dataset_name} {method} seed={seed}] already complete -> skipping")
        return {}
    os.makedirs(out_dir, exist_ok=True)

    split = prepare_seed_split(spec.dataframe, spec.target_column, spec.categorical_cols, spec.numerical_cols,
                               spec.dataset_name, seed=seed, test_size=args.test_size,
                               target_mapping=spec.target_mapping)
    fp = split["preprocessor"]
    X_train, y_train, X_test, y_test = split["X_train"], split["y_train"], split["X_test"], split["y_test"]
    write_json(split_fingerprint(split["train_df"], split["test_df"], split["train_row_ids"], split["test_row_ids"]), os.path.join(out_dir, "split_fingerprint.json"))
    if args.export_splits:
        export_split(split, os.path.join(args.export_splits, spec.dataset_name, f"seed{seed}"), spec.target_column)

    n_syn = args.n_synthetic or len(y_train)
    counts = target_counts(y_train, n_syn, args.synthetic_class_ratio)
    inv_map = {v: k for k, v in (spec.target_mapping or {0: 0, 1: 1}).items()}
    protocol = "TSTR"
    t0 = time.time()
    if key == "REAL":
        X_syn, y_syn, params = X_train.copy(), y_train.copy(), {"note": "real training partition (TRTR reference)"}
        protocol = "TRTR"
    elif key in {"SMOTE", "ADASYN"}:
        X_syn, y_syn, params = oversampler_synthetic(key, X_train, y_train, counts, fp.categorical_slices,
                                                     seed, args.k_neighbors)
    elif key in SDV_METHODS:
        counts_raw = {str(inv_map[c]): n for c, n in counts.items()}
        fake_raw, params = sdv_synthetic(key, split["train_df"], spec.target_column, counts_raw, seed,
                                         args.sdv_epochs, args.sdv_batch_size)
        X_syn, y_syn = transform_with_preprocessor(fake_raw, fp)
    elif key == "EXTERNAL":
        path = os.path.join(args.external_dir, spec.dataset_name, f"seed{seed}", f"{method}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"external synthetic file not found: {path}")
        fake_raw = pd.read_csv(path)
        X_syn, y_syn = transform_with_preprocessor(fake_raw, fp)
        params = {"source_file": path, "note": "generated outside this package from the exported train split"}
    else:
        raise ValueError(f"unknown method {method_arg}")
    fit_and_sample_sec = time.time() - t0

    fake_df = inverse_transform(X_syn, fp)
    fake_df.assign(**{spec.target_column: np.asarray(y_syn).astype(int)}).to_csv(
        os.path.join(out_dir, "synthetic_decoded.csv"), index=False)
    write_json({"method": method, "seed": seed, "test_size": args.test_size, "n_synthetic": int(len(y_syn)),
                "synthetic_class_ratio": args.synthetic_class_ratio,
                "generated_counts": {int(c): int((np.asarray(y_syn).astype(int) == c).sum()) for c in (0, 1)},
                "preprocessing": "identical train-fitted preprocessor as CD-GAN (preprocessing.prepare_seed_split)",
                "tuning": "none: library defaults except the values listed here",
                "hyperparameters": params, "versions": package_versions()},
               os.path.join(out_dir, "run_config.json"))
    write_json({"fit_and_sample_time_sec": fit_and_sample_sec, "training_time_sec": fit_and_sample_sec,
                "epochs": args.sdv_epochs if key in {"CTGAN", "TVAE", "COPULAGAN"} else None},
               os.path.join(out_dir, "runtime.json"))

    summary = evaluate_synthetic(
        method=method, dataset=spec.dataset_name, seed=seed, out_dir=out_dir, fp=fp,
        real_train_df=impute_dataframe(split["train_df"], fp), y_train_real=y_train, X_train_real=X_train,
        X_test_real=X_test, y_test_real=y_test, fake_df=fake_df, X_syn=X_syn, y_syn=y_syn,
        utility_protocol=protocol,
    )
    print(f"[{spec.dataset_name} {method} seed={seed}] F1={summary['util_f1']:.4f} "
          f"PR-AUC={summary['util_pr_auc']:.4f} WDnum={summary['num_wd_norm_mean']:.4f} "
          f"TVcat={summary['cat_tv_distance_mean']:.4f} ({fit_and_sample_sec:.1f}s)")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-synthetic", type=int, default=None)
    ap.add_argument("--synthetic-class-ratio", choices=["train", "balanced"], default="train")
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument("--sdv-epochs", type=int, default=300)
    ap.add_argument("--sdv-batch-size", type=int, default=500)
    ap.add_argument("--external-dir", default="external_synth")
    ap.add_argument("--export-splits", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    spec = load_dataset(args.dataset, args.data_path)
    for m in args.methods:
        for seed in args.seeds:
            run_one(args, spec, m, seed)


if __name__ == "__main__":
    main()
