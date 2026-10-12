"""
quality_metrics.py -- Rev 4: ONE evaluator for CD-GAN and every baseline
(reviewer points 2, 16, 17, 19).

Torch-free. Everything here consumes arrays/dataframes that were produced
under the same leak-free split (preprocessing fit on the real TRAIN
partition only), so CD-GAN and baselines are scored by identical code.

Protocols (unchanged from Rev 3, now enforced in one place):
  (A) fidelity / dependency / diversity / detection : real TRAIN vs synthetic
  (B) utility (TSTR)                                : train on synthetic,
                                                      test on untouched real TEST
  (C) privacy                                       : real TRAIN vs synthetic

What is new in Rev 4
--------------------
* Utility: binary F1 (positive = label 1), macro-F1, balanced accuracy, MCC,
  and minority-class precision / recall / F1 / PR-AUC, where the minority
  label is taken from the real TRAIN partition (never the test set).
* Type-aware fidelity: numerical columns -> normalized Wasserstein distance
  and KS statistic; categorical columns -> total-variation distance and
  JSD. Wasserstein distance is NEVER computed on categorical columns.
* Dependency: mixed-type association-matrix MAE (Pearson / correlation
  ratio / point-biserial / Cramer's V / phi, from evaluation.py), the same
  error computed within each class (class-conditional), and the Frobenius
  distance between numerical Pearson correlation matrices.
* Detection: cross-validated real-vs-synthetic classifier AUC (logistic
  regression and a depth-limited gradient-boosting model); 0.5 means
  indistinguishable. Reported with ``detection_auc_gap = 2*|AUC-0.5|``.
* Diversity / mode collapse: unique-row ratio and within-synthetic
  duplicate ratio (decoded space), category coverage and rare-category
  coverage, nearest-neighbour diversity ratio (syn->syn vs real->real
  nearest-neighbour distance), plus density/coverage.
* Minority-class quality: numerical/categorical fidelity, association error
  and coverage recomputed on minority-class rows only.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from evaluation import (
    build_classifiers,
    categorical_quality_report,
    density_coverage,
    pairwise_relationship_matrix,
    privacy_diagnostics,
    statistical_quality_report,
)

UTILITY_METRICS = [
    "f1", "f1_macro", "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
    "minority_precision", "minority_recall", "minority_f1", "minority_pr_auc",
]


# ======================================================================
# Utility (TSTR)
# ======================================================================
def minority_label_from_train(y_train_real: np.ndarray) -> int:
    """Minority label = the less frequent class in the REAL TRAIN partition.
    For an exactly balanced training set, label 1 is used."""
    y = np.asarray(y_train_real).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"binary target expected, found classes {classes.tolist()}")
    if counts[0] == counts[1]:
        return 1
    return int(classes[int(np.argmin(counts))])


def extended_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                            minority_label: int) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "minority_precision": float(precision_score(y_true, y_pred, pos_label=minority_label, zero_division=0)),
        "minority_recall": float(recall_score(y_true, y_pred, pos_label=minority_label, zero_division=0)),
        "minority_f1": float(f1_score(y_true, y_pred, pos_label=minority_label, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    try:
        score_min = y_score if minority_label == 1 else 1.0 - y_score
        out["minority_pr_auc"] = float(average_precision_score((y_true == minority_label).astype(int), score_min))
    except ValueError:
        out["minority_pr_auc"] = float("nan")
    return out


def evaluate_utility_extended(X_train_clf: np.ndarray, y_train_clf: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray, seed: int,
                              minority_label: int, protocol: str = "TSTR") -> pd.DataFrame:
    """Same six classifiers as Rev 3 (evaluation.build_classifiers), more
    metrics. ``protocol`` is a label recorded in every row (TSTR, TRTR,
    augmented, or resampled for SMOTE-style oversamplers)."""
    rows = []
    if len(np.unique(np.asarray(y_train_clf).astype(int))) < 2:
        warnings.warn("classifier training set contains a single class; utility set to NaN")
        for name in build_classifiers(seed):
            rows.append({"classifier": name, "protocol": protocol, **{m: float("nan") for m in UTILITY_METRICS}})
        return pd.DataFrame(rows)
    for name, clf in build_classifiers(seed).items():
        clf.fit(X_train_clf, np.asarray(y_train_clf).astype(int))
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
        rows.append({"classifier": name, "protocol": protocol,
                     **extended_binary_metrics(y_test, y_pred, y_score, minority_label)})
    return pd.DataFrame(rows)


# ======================================================================
# Fidelity (type-aware)
# ======================================================================
def type_aware_fidelity(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                        numerical_cols: List[str], categorical_cols: List[str],
                        numerical_percentile_bounds: Dict, categorical_categories: Dict
                        ) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    num_df = statistical_quality_report(real_df, fake_df, numerical_cols,
                                        numerical_percentile_bounds=numerical_percentile_bounds)
    cat_df = categorical_quality_report(real_df, fake_df, categorical_cols,
                                        categorical_categories=categorical_categories)
    summary = {
        "n_numerical_cols": len(numerical_cols),
        "n_categorical_cols": len(categorical_cols),
        "num_wd_norm_mean": float(num_df["wasserstein_normalized"].mean()) if len(num_df) else float("nan"),
        "num_wd_norm_median": float(num_df["wasserstein_normalized"].median()) if len(num_df) else float("nan"),
        "num_ks_stat_mean": float(num_df["ks_stat"].mean()) if len(num_df) else float("nan"),
        "cat_tv_distance_mean": float(cat_df["tv_distance"].mean()) if len(cat_df) else float("nan"),
        "cat_jsd_mean": float(cat_df["categorical_jsd"].mean()) if len(cat_df) else float("nan"),
    }
    return summary, num_df, cat_df


# ======================================================================
# Dependency preservation
# ======================================================================
def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def _assoc_mae(real_df, fake_df, num_cols, cat_cols, binary_orders) -> float:
    if len(num_cols) + len(cat_cols) < 2 or len(real_df) < 3 or len(fake_df) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rm = pairwise_relationship_matrix(real_df, num_cols, cat_cols, binary_orders).to_numpy(dtype=float)
        fm = pairwise_relationship_matrix(fake_df, num_cols, cat_cols, binary_orders).to_numpy(dtype=float)
    iu = np.triu_indices(rm.shape[0], k=1)
    a, b = rm[iu], fm[iu]
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float("nan")


def dependency_metrics(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                       y_real: np.ndarray, y_fake: np.ndarray,
                       numerical_cols: List[str], categorical_cols: List[str],
                       binary_category_orders: Dict, max_rows: int = 5000, seed: int = 0
                       ) -> Dict[str, float]:
    """Association-matrix MAE (all rows and within each class) and the
    Frobenius distance between numerical Pearson correlation matrices.
    Rows are subsampled to ``max_rows`` (seeded) purely for speed."""
    r = real_df.assign(_y=np.asarray(y_real).astype(int)).reset_index(drop=True)
    f = fake_df.assign(_y=np.asarray(y_fake).astype(int)).reset_index(drop=True)
    r_s, f_s = _subsample(r, max_rows, seed), _subsample(f, max_rows, seed)
    out = {"assoc_mae": _assoc_mae(r_s, f_s, numerical_cols, categorical_cols, binary_category_orders)}

    per_class, weights = [], []
    for cls in sorted(r["_y"].unique()):
        rc = _subsample(r[r["_y"] == cls], max_rows, seed)
        fc = _subsample(f[f["_y"] == cls], max_rows, seed)
        val = _assoc_mae(rc, fc, numerical_cols, categorical_cols, binary_category_orders)
        out[f"assoc_mae_class{int(cls)}"] = val
        if not np.isnan(val):
            per_class.append(val)
            weights.append((r["_y"] == cls).mean())
    out["assoc_mae_class_conditional"] = (
        float(np.average(per_class, weights=weights)) if per_class else float("nan"))

    if len(numerical_cols) >= 2:
        rc = r_s[numerical_cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
        fc = f_s[numerical_cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
        diff = np.nan_to_num(rc - fc)
        p = len(numerical_cols)
        out["num_pearson_frobenius"] = float(np.linalg.norm(diff, "fro"))
        out["num_pearson_mae"] = float(np.abs(diff[np.triu_indices(p, 1)]).mean())
    else:
        out["num_pearson_frobenius"] = float("nan")
        out["num_pearson_mae"] = float("nan")
    return out


# ======================================================================
# Detection (real vs synthetic)
# ======================================================================
def detection_metrics(X_real: np.ndarray, X_syn: np.ndarray, seed: int, max_rows: int = 5000,
                      n_folds: int = 5) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = int(min(len(X_real), len(X_syn), max_rows))
    if n < 2 * n_folds:
        return {"detection_auc_lr": float("nan"), "detection_auc_gbm": float("nan"),
                "detection_auc_gap_lr": float("nan"), "detection_auc_gap_gbm": float("nan")}
    xr = X_real[rng.choice(len(X_real), n, replace=False)]
    xs = X_syn[rng.choice(len(X_syn), n, replace=False)]
    X = np.vstack([xr, xs])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = {}
    models = {
        "lr": LogisticRegression(max_iter=2000),
        "gbm": HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=seed),
    }
    for key, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, p))
        out[f"detection_auc_{key}"] = auc
        out[f"detection_auc_gap_{key}"] = float(2 * abs(auc - 0.5))
    return out


# ======================================================================
# Diversity / mode collapse
# ======================================================================
def _row_keys(df: pd.DataFrame, numerical_cols: List[str], decimals: int = 6) -> pd.Series:
    d = df.copy()
    for c in numerical_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").round(decimals)
    return pd.util.hash_pandas_object(d.astype(str), index=False)


def _mean_nn_distance(X: np.ndarray, max_rows: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        X = X[rng.choice(len(X), max_rows, replace=False)]
    if len(X) < 3:
        return float("nan")
    d, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    return float(np.mean(d[:, 1]))


def diversity_metrics(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                      X_real: np.ndarray, X_syn: np.ndarray,
                      numerical_cols: List[str], categorical_cols: List[str],
                      categorical_categories: Dict, seed: int = 0, max_rows: int = 5000,
                      rare_threshold: float = 0.05) -> Dict[str, float]:
    cols = categorical_cols + numerical_cols
    syn_keys = _row_keys(fake_df[cols], numerical_cols)
    real_keys = _row_keys(real_df[cols], numerical_cols)
    out = {
        "syn_unique_row_ratio": float(syn_keys.nunique() / max(len(syn_keys), 1)),
        "real_unique_row_ratio": float(real_keys.nunique() / max(len(real_keys), 1)),
    }
    out["syn_duplicate_ratio"] = 1.0 - out["syn_unique_row_ratio"]

    cov, rare_cov = [], []
    for c in categorical_cols:
        universe = [str(v) for v in (categorical_categories or {}).get(c, sorted(real_df[c].astype(str).unique()))]
        real_freq = real_df[c].astype(str).value_counts(normalize=True)
        present = set(fake_df[c].astype(str).unique())
        cov.append(np.mean([u in present for u in universe]))
        rare = [u for u in universe if real_freq.get(u, 0.0) < rare_threshold and real_freq.get(u, 0.0) > 0]
        if rare:
            rare_cov.append(np.mean([u in present for u in rare]))
    out["category_coverage"] = float(np.mean(cov)) if cov else float("nan")
    out["rare_category_coverage"] = float(np.mean(rare_cov)) if rare_cov else float("nan")

    d_real = _mean_nn_distance(X_real, max_rows, seed)
    d_syn = _mean_nn_distance(X_syn, max_rows, seed)
    out["nn_dist_real_real"] = d_real
    out["nn_dist_syn_syn"] = d_syn
    out["nn_diversity_ratio"] = float(d_syn / d_real) if d_real and d_real > 0 else float("nan")
    rng = np.random.default_rng(seed)
    xr = X_real if len(X_real) <= max_rows else X_real[rng.choice(len(X_real), max_rows, replace=False)]
    xs = X_syn if len(X_syn) <= max_rows else X_syn[rng.choice(len(X_syn), max_rows, replace=False)]
    out.update(density_coverage(xr, xs))
    return out


# ======================================================================
# Master evaluator
# ======================================================================
def evaluate_synthetic(
    *,
    method: str,
    dataset: str,
    seed: int,
    out_dir: str,
    fp,
    real_train_df: pd.DataFrame,
    y_train_real: np.ndarray,
    X_train_real: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    fake_df: pd.DataFrame,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    utility_protocol: str = "TSTR",
    X_clf_train: Optional[np.ndarray] = None,
    y_clf_train: Optional[np.ndarray] = None,
    max_rows: int = 5000,
    reference_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Scores one synthetic dataset and writes:
        utility.csv, fidelity_numerical.csv, fidelity_categorical.csv,
        quality_summary.csv (one flat row: every scalar metric).

    ``fake_df``  : decoded synthetic predictors (original column names).
    ``X_syn``    : the same synthetic rows in the TRAIN-fitted encoded space
                   (categorical blocks one-hot, numericals standardized).
    ``X_clf_train``/``y_clf_train``: classifier training set if it differs
                   from (X_syn, y_syn) -- e.g. SMOTE's resampled set. The
                   label ``utility_protocol`` is stored with every row.
    ``reference_mask``: optional boolean mask over the real TRAIN rows used
                   as the reference for fidelity / dependency / detection /
                   diversity / privacy. Used for SMOTE-style oversamplers,
                   whose generated rows belong to the minority class only:
                   they are compared with real rows of the same class
                   (class-matched), not with the whole training set.
                   Utility and the minority label always use all rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_cols, cat_cols = list(fp.numerical_cols), list(fp.categorical_cols)
    cat_universe = dict(zip(fp.categorical_cols, fp.categorical_categories))
    minority = minority_label_from_train(y_train_real)
    tag = {"method": method, "dataset": dataset, "seed": seed}
    y_all_real = np.asarray(y_train_real)
    if reference_mask is not None:
        reference_mask = np.asarray(reference_mask, dtype=bool)
        real_train_df = real_train_df[reference_mask].reset_index(drop=True)
        y_train_real = y_all_real[reference_mask]
        X_train_real = X_train_real[reference_mask]

    Xc = X_syn if X_clf_train is None else X_clf_train
    yc = y_syn if y_clf_train is None else y_clf_train
    util = evaluate_utility_extended(Xc, yc, X_test_real, y_test_real, seed, minority, utility_protocol)
    for i, (k, v) in enumerate(tag.items()):
        util.insert(i, k, v)
    util["minority_label"] = minority
    util.to_csv(os.path.join(out_dir, "utility.csv"), index=False)

    fid, num_df, cat_df = type_aware_fidelity(real_train_df, fake_df, num_cols, cat_cols,
                                              fp.numerical_percentile_bounds, cat_universe)
    for d in (num_df, cat_df):
        for i, (k, v) in enumerate(tag.items()):
            d.insert(i, k, v)
    num_df.to_csv(os.path.join(out_dir, "fidelity_numerical.csv"), index=False)
    cat_df.to_csv(os.path.join(out_dir, "fidelity_categorical.csv"), index=False)

    dep = dependency_metrics(real_train_df, fake_df, y_train_real, y_syn, num_cols, cat_cols,
                             fp.binary_category_orders, max_rows=max_rows, seed=seed)
    det = detection_metrics(X_train_real, X_syn, seed, max_rows=max_rows)
    div = diversity_metrics(real_train_df, fake_df, X_train_real, X_syn, num_cols, cat_cols,
                            cat_universe, seed=seed, max_rows=max_rows)

    # minority-class-only quality
    rm = np.asarray(y_train_real).astype(int) == minority
    fm = np.asarray(y_syn).astype(int) == minority
    mino: Dict[str, float] = {"minority_label": minority,
                              "n_real_minority": int(rm.sum()), "n_syn_minority": int(fm.sum())}
    if rm.sum() >= 5 and fm.sum() >= 5:
        f_min, _, _ = type_aware_fidelity(real_train_df[rm].reset_index(drop=True),
                                          fake_df[fm].reset_index(drop=True),
                                          num_cols, cat_cols, fp.numerical_percentile_bounds, cat_universe)
        mino.update({f"minority_{k}": v for k, v in f_min.items() if not k.startswith("n_")})
        mino["minority_assoc_mae"] = _assoc_mae(
            _subsample(real_train_df[rm], max_rows, seed), _subsample(fake_df[fm], max_rows, seed),
            num_cols, cat_cols, fp.binary_category_orders)
        dc = density_coverage(X_train_real[rm][:max_rows], X_syn[fm][:max_rows])
        mino["minority_density"], mino["minority_coverage"] = dc["density"], dc["coverage"]

    priv = privacy_diagnostics(X_train_real, X_syn)
    util_mean = util[UTILITY_METRICS].mean(numeric_only=True).to_dict()

    row = {**tag, "utility_protocol": utility_protocol,
           "fidelity_reference": "class_matched_train_rows" if reference_mask is not None else "all_train_rows",
           "n_synthetic": int(len(y_syn)),
           "synthetic_class1_share": float(np.mean(np.asarray(y_syn).astype(int) == 1)),
           **{f"util_{k}": v for k, v in util_mean.items()},
           **fid, **dep, **det, **div, **mino, **priv}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "quality_summary.csv"), index=False)
    return row
