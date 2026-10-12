"""
evaluation.py -- downstream utility (TSTR) + statistical/relationship
quality metrics (Sections 11-13).

Two DISTINCT kinds of metric live in this file, and they are never
confused with each other or with the training-time covariance regularizer
in model.py:

  1. Downstream classifier utility (Section 11-12): TSTR by default,
     "augmented" as an explicit secondary mode.
  2. Statistical / distributional quality metrics computed in the
     ORIGINAL (decoded) data space (Section 13): these are diagnostics
     reported after training, not anything used inside L_G or L_D.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except Exception:
    _HAS_LIGHTGBM = False


# ======================================================================
# Section 12: downstream classifiers
# ======================================================================
def build_classifiers(seed: int) -> Dict[str, object]:
    clfs = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }
    if _HAS_LIGHTGBM:
        clfs["LightGBM"] = LGBMClassifier(random_state=seed, verbosity=-1)
    else:
        warnings.warn("lightgbm not installed; skipping LightGBM classifier.")
    return clfs


def _safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out = {"f1": float(f1_score(y_true, y_pred, zero_division=0))}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def evaluate_tstr_per_classifier(
    X_train_clf: np.ndarray,
    y_train_clf: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    seed: int,
    evaluation_mode: str = "TSTR",
) -> pd.DataFrame:
    """Section 11-12.

    evaluation_mode='TSTR' (DEFAULT): train on (X_train_clf, y_train_clf)
        which the caller must set to the SYNTHETIC training set only, test
        on the untouched real holdout.
    evaluation_mode='augmented' (SECONDARY, opt-in): caller passes the
        already-concatenated [real_train ; synthetic] arrays.

    This function itself does not decide which arrays to concatenate -- it
    only trains/evaluates on what it is given and records which
    `evaluation_mode` label was requested, so the two protocols can never
    be silently confused downstream.
    """
    if evaluation_mode not in {"TSTR", "augmented"}:
        raise ValueError("evaluation_mode must be 'TSTR' or 'augmented'.")

    rows = []
    for clf_name, clf in build_classifiers(seed).items():
        clf.fit(X_train_clf, y_train_clf)
        y_pred = clf.predict(X_test_real)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_real)[:, 1]
        else:
            y_score = clf.decision_function(X_test_real)
        m = _safe_binary_metrics(y_test_real, y_pred, y_score)
        rows.append({"classifier": clf_name, "evaluation_mode": evaluation_mode, **m})
    return pd.DataFrame(rows)


def build_tstr_training_set(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    evaluation_mode: str,
    X_train_real: Optional[np.ndarray] = None,
    y_train_real: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the classifier TRAINING set for the requested protocol.

    TSTR (default): X_train_clf, y_train_clf = X_syn, y_syn  (real data never
        enters classifier training).
    augmented (secondary): np.vstack([X_train_real, X_syn]) -- requires
        X_train_real/y_train_real to be supplied explicitly, so this can
        never happen silently.
    """
    if evaluation_mode == "TSTR":
        return X_syn, y_syn
    if evaluation_mode == "augmented":
        if X_train_real is None or y_train_real is None:
            raise ValueError("augmented mode requires X_train_real/y_train_real.")
        X = np.vstack([X_train_real, X_syn])
        y = np.concatenate([y_train_real, y_syn])
        return X, y
    raise ValueError("evaluation_mode must be 'TSTR' or 'augmented'.")


# ======================================================================
# Section 13a: univariate statistical quality metrics
# ======================================================================
def _hist_prob(a: np.ndarray, b: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa.astype(np.float64) + 1e-8
    pb = pb.astype(np.float64) + 1e-8
    return pa / pa.sum(), pb / pb.sum()


def numerical_column_metrics(
    real_col: np.ndarray,
    fake_col: np.ndarray,
    norm_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """KLD/JSD/KS + BOTH raw and normalized Wasserstein distance.

    ``wasserstein_raw`` is on the original column scale (can be dominated
    by large-scale columns when averaged across heterogeneous columns).
    ``wasserstein_normalized`` rescales both real and fake values by a
    robust min-max derived from ``norm_bounds`` = (p1, p99) of the REAL
    TRAINING distribution (never the test set) before computing WD, so it
    is comparable across columns/datasets. NaN if ``norm_bounds`` is not
    supplied.
    """
    pa, pb = _hist_prob(real_col, fake_col)
    kld = float(entropy(pa, pb))
    jsd = float(0.5 * entropy(pa, (pa + pb) / 2) + 0.5 * entropy(pb, (pa + pb) / 2))
    wd_raw = float(wasserstein_distance(real_col, fake_col))
    ks_stat, ks_p = ks_2samp(real_col, fake_col)

    wd_norm = float("nan")
    if norm_bounds is not None:
        p1, p99 = norm_bounds
        scale = (p99 - p1) if (p99 - p1) > 1e-12 else 1e-12
        real_norm = (real_col - p1) / scale
        fake_norm = (fake_col - p1) / scale
        wd_norm = float(wasserstein_distance(real_norm, fake_norm))

    return {
        "kld": kld,
        "jsd": jsd,
        "wasserstein_raw": wd_raw,
        "wasserstein_normalized": wd_norm,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def statistical_quality_report(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    numerical_cols: List[str],
    numerical_percentile_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Per-numerical-column KLD/JSD/raw+normalized-Wasserstein/KS (Section 13).

    ``numerical_percentile_bounds`` should be
    ``FittedPreprocessor.numerical_percentile_bounds`` (TRAIN-derived 1st/
    99th percentiles) -- pass it to get ``wasserstein_normalized``;
    omitting it still returns ``wasserstein_raw`` (and NaN for the
    normalized column) rather than failing.
    """
    rows = []
    for col in numerical_cols:
        bounds = (numerical_percentile_bounds or {}).get(col)
        m = numerical_column_metrics(
            pd.to_numeric(real_df[col], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(fake_df[col], errors="coerce").dropna().to_numpy(),
            norm_bounds=bounds,
        )
        rows.append({"column": col, **m})
    return pd.DataFrame(rows)


# ======================================================================
# Section 16: categorical univariate quality metrics (EVALUATION ONLY --
# never fed into GAN training)
# ======================================================================
def categorical_column_metrics(
    real_col: pd.Series,
    fake_col: pd.Series,
    category_universe: Optional[List[str]] = None,
) -> Dict[str, float]:
    """JSD + total-variation distance (and its complement) between the
    real and synthetic category-probability vectors for ONE categorical
    column.

    ``category_universe`` should be the TRAIN-fitted category list for
    this column (e.g. ``FittedPreprocessor.categorical_categories`` in
    ``categorical_cols`` order) so a category missing from one side still
    gets an explicit zero probability rather than silently vanishing from
    the comparison. Falls back to the union of observed categories if not
    supplied.
    """
    real_col = real_col.astype(str)
    fake_col = fake_col.astype(str)
    universe = category_universe or sorted(set(real_col.unique()) | set(fake_col.unique()))

    real_counts = real_col.value_counts().reindex(universe, fill_value=0).to_numpy(dtype=np.float64)
    fake_counts = fake_col.value_counts().reindex(universe, fill_value=0).to_numpy(dtype=np.float64)

    pa = (real_counts + 1e-8) / (real_counts.sum() + 1e-8 * len(universe))
    pb = (fake_counts + 1e-8) / (fake_counts.sum() + 1e-8 * len(universe))

    jsd = float(0.5 * entropy(pa, (pa + pb) / 2) + 0.5 * entropy(pb, (pa + pb) / 2))
    tv_distance = float(0.5 * np.abs(pa - pb).sum())
    return {"categorical_jsd": jsd, "tv_distance": tv_distance, "tv_complement": 1.0 - tv_distance}


def categorical_quality_report(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    categorical_cols: List[str],
    categorical_categories: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Per-categorical-column JSD/TV-distance/TV-complement (Section 16).

    ``categorical_categories`` should be
    ``{col: FittedPreprocessor.categorical_categories[i]}`` (TRAIN-fitted
    category universe per column).
    """
    rows = []
    for col in categorical_cols:
        universe = (categorical_categories or {}).get(col)
        m = categorical_column_metrics(real_df[col], fake_df[col], category_universe=universe)
        rows.append({"column": col, **m})
    return pd.DataFrame(rows)


# ======================================================================
# Section 13b: mixed-type pairwise relationship measures (ORIGINAL space)
# ======================================================================
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta (correlation ratio) for a numerical column explained by a
    categorical column."""
    values = pd.to_numeric(values, errors="coerce")
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df["cat"].nunique() < 2 or len(df) < 2:
        return float("nan")
    grand_mean = df["val"].mean()
    ss_between = df.groupby("cat")["val"].apply(lambda v: len(v) * (v.mean() - grand_mean) ** 2).sum()
    ss_total = ((df["val"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def point_biserial(binary: pd.Series, values: pd.Series, mapping: Dict) -> float:
    """Point-biserial correlation with an EXPLICIT, caller-supplied
    category->{0,1} mapping.

    Section 17 fix: the previous implementation inferred ``uniq[0]->0,
    uniq[1]->1`` from ``pandas.Series.unique()``, whose order is
    "first appearance in THIS series" -- real and synthetic data can
    (and empirically will) see their two category levels in different
    first-appearance order, flipping the sign of the correlation between
    the real-data and synthetic-data computations even when the
    underlying relationship is identical. ``mapping`` must now be
    supplied by the caller and be the SAME dict for both the real and the
    synthetic computation of the same column (see
    ``FittedPreprocessor.binary_category_orders``, derived once from
    TRAIN data). This function no longer infers its own mapping.
    """
    values = pd.to_numeric(values, errors="coerce")
    df = pd.DataFrame({"b": binary.astype(str), "v": values}).dropna()
    unseen = set(df["b"].unique()) - set(mapping.keys())
    if unseen:
        # A category outside the known train-fitted universe (should not
        # happen for projected synthetic data or in-sample real data, but
        # fail loudly rather than silently mis-scoring if it does).
        raise ValueError(f"point_biserial: categories {unseen} not in the supplied mapping {mapping}")
    if df["b"].nunique() < 2 or len(df) < 2:
        return float("nan")
    b_num = df["b"].map(mapping).to_numpy()
    r, _ = stats.pointbiserialr(b_num, df["v"].to_numpy())
    return float(r)


def phi_coefficient(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape != (2, 2):
        return cramers_v(x, y)  # fall back if not strictly 2x2
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / n)) if n > 0 else float("nan")


def pairwise_relationship_matrix(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    binary_category_orders: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    """One appropriate association measure per column pair, in the
    ORIGINAL (decoded) data space:

        numerical-numerical    : Pearson correlation
        numerical-categorical  : correlation ratio / eta
        numerical-binary       : point-biserial (DETERMINISTIC mapping)
        categorical-categorical: Cramer's V
        binary-binary           : phi coefficient

    ``binary_category_orders`` (Section 17 fix): pass
    ``FittedPreprocessor.binary_category_orders`` -- a
    {categorical_col: [level0, level1]} mapping computed ONCE from TRAIN
    data. "Is this categorical column binary" is then decided by
    membership in this dict (a fixed, train-derived fact), NOT by
    re-counting ``nunique()`` on whatever subset of rows this particular
    call happens to see -- real and synthetic data can otherwise disagree
    on whether a column "looks binary" in a given sample. The SAME dict
    must be passed for both the real-data and the fake-data call so
    ``pairwise_relationship_error`` compares like with like.
    """
    binary_category_orders = binary_category_orders or {}
    cols = numerical_cols + categorical_cols
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                mat.loc[c1, c2] = 1.0
                continue
            c1_num, c2_num = c1 in numerical_cols, c2 in numerical_cols
            if c1_num and c2_num:
                v1 = pd.to_numeric(df[c1], errors="coerce")
                v2 = pd.to_numeric(df[c2], errors="coerce")
                mat.loc[c1, c2] = float(v1.corr(v2))
            elif c1_num and not c2_num:
                if c2 in binary_category_orders:
                    mapping = {binary_category_orders[c2][0]: 0, binary_category_orders[c2][1]: 1}
                    mat.loc[c1, c2] = point_biserial(df[c2], df[c1], mapping)
                else:
                    mat.loc[c1, c2] = correlation_ratio(df[c2], df[c1])
            elif c2_num and not c1_num:
                if c1 in binary_category_orders:
                    mapping = {binary_category_orders[c1][0]: 0, binary_category_orders[c1][1]: 1}
                    mat.loc[c1, c2] = point_biserial(df[c1], df[c2], mapping)
                else:
                    mat.loc[c1, c2] = correlation_ratio(df[c1], df[c2])
            else:
                mat.loc[c1, c2] = (
                    phi_coefficient(df[c1], df[c2])
                    if (c1 in binary_category_orders and c2 in binary_category_orders)
                    else cramers_v(df[c1], df[c2])
                )
    return mat


def pairwise_relationship_error(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    binary_category_orders: Optional[Dict[str, List]] = None,
) -> Dict[str, float]:
    """Mean absolute error between the real and synthetic pairwise
    relationship matrices (upper triangle only). Pass
    ``FittedPreprocessor.binary_category_orders`` so both matrices use an
    identical, train-derived binary/point-biserial convention (Section 17)."""
    real_mat = pairwise_relationship_matrix(real_df, numerical_cols, categorical_cols, binary_category_orders)
    fake_mat = pairwise_relationship_matrix(fake_df, numerical_cols, categorical_cols, binary_category_orders)
    iu = np.triu_indices(len(real_mat), k=1)
    real_vals = real_mat.to_numpy()[iu]
    fake_vals = fake_mat.to_numpy()[iu]
    mask = ~(np.isnan(real_vals) | np.isnan(fake_vals))
    if mask.sum() == 0:
        return {"pairwise_mae": float("nan"), "n_pairs": 0}
    mae = float(np.mean(np.abs(real_vals[mask] - fake_vals[mask])))
    return {"pairwise_mae": mae, "n_pairs": int(mask.sum())}


# ======================================================================
# Section 13c: density & coverage (encoded space, k-NN based)
# ======================================================================
def density_coverage(
    real_encoded: np.ndarray,
    fake_encoded: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """k-NN-sphere density & coverage (Naeem et al., 2020 style), computed
    on the encoded feature space. Both are diagnostics of how well the
    synthetic distribution fills / stays within the real manifold; neither
    is used anywhere in training."""
    n_real = real_encoded.shape[0]
    n_fake = fake_encoded.shape[0]
    if n_real < k + 1 or n_fake < 1:
        return {"density": float("nan"), "coverage": float("nan")}

    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real_encoded)
    real_dists, _ = nn_real.kneighbors(real_encoded)
    real_radii = real_dists[:, -1]  # distance to k-th nearest OTHER real point

    dist_fake_to_real = cdist(fake_encoded, real_encoded)  # [n_fake, n_real]

    # density: for each fake point, how many real spheres contain it, averaged
    within_sphere = dist_fake_to_real < real_radii[None, :]
    density = float(within_sphere.sum() / (k * n_fake))

    # coverage: fraction of real points whose sphere contains >=1 fake point
    covered = within_sphere.any(axis=0)
    coverage = float(covered.mean())

    return {"density": density, "coverage": coverage}


# ======================================================================
# Section 18: privacy / memorization diagnostics (EVALUATION ONLY)
# ======================================================================
def privacy_diagnostics(
    X_train_real: np.ndarray,
    X_syn: np.ndarray,
    chunk_size: int = 1000,
    near_copy_percentile: float = 1.0,
    exact_copy_eps: float = 1e-6,
) -> Dict[str, float]:
    """Empirical privacy/memorization diagnostics, REAL TRAINING vs
    SYNTHETIC data (never real test data -- test-set proximity is an
    evaluation-fairness question, not a privacy one).

    Computed in the encoded space (post categorical-projection synthetic
    data, i.e. pass ``X_syn_projected_encoded``, against the real training
    encoded data, so distances between one-hot categorical blocks are
    directly comparable).

    Returns
    -------
    dcr_mean / dcr_median / dcr_min : Distance to Closest real Record --
        for every synthetic row, its Euclidean distance to the NEAREST
        real training row. Small values across many synthetic rows
        suggest the generator is reproducing near-copies of real records.
    nndr_mean / nndr_median : Nearest-Neighbour Distance Ratio -- for
        every synthetic row, (distance to nearest real row) / (distance
        to 2nd-nearest real row). Values near 0 mean the synthetic row
        sits suspiciously close to exactly one real record relative to
        its surrounding neighbourhood; values near 1 mean it is roughly
        equidistant to several real records (ordinary, non-memorized
        behaviour).
    exact_copy_ratio : fraction of synthetic rows whose DCR is below
        ``exact_copy_eps`` (a literal encoded-space duplicate of some
        real training row).
    near_copy_ratio : fraction of synthetic rows whose DCR falls below a
        DATA-DRIVEN threshold -- the ``near_copy_percentile``-th
        percentile of real-to-real nearest-neighbour distances (computed
        on X_train_real itself), i.e. "closer to a real row than real
        rows typically are to each other."
    near_copy_threshold : the resolved distance threshold used above (so
        it's auditable, not a magic number).

    These are EMPIRICAL diagnostics only -- no claim of formal
    differential privacy is made anywhere in this function or its
    callers.
    """
    n_real = X_train_real.shape[0]
    n_syn = X_syn.shape[0]
    if n_real < 2 or n_syn < 1:
        return {k: float("nan") for k in [
            "dcr_mean", "dcr_median", "dcr_min", "nndr_mean", "nndr_median",
            "exact_copy_ratio", "near_copy_ratio", "near_copy_threshold",
        ]}

    # Data-driven near-copy threshold: real-to-real nearest-neighbour
    # distance distribution (excluding each point's zero self-distance).
    nn_real_real = NearestNeighbors(n_neighbors=2).fit(X_train_real)
    real_real_dists, _ = nn_real_real.kneighbors(X_train_real)
    real_nn_dists = real_real_dists[:, 1]  # nearest OTHER real row
    near_copy_threshold = float(np.percentile(real_nn_dists, near_copy_percentile))

    nn2 = NearestNeighbors(n_neighbors=2).fit(X_train_real)

    dcr_all = np.empty(n_syn, dtype=np.float64)
    nndr_all = np.empty(n_syn, dtype=np.float64)
    for start in range(0, n_syn, chunk_size):
        chunk = X_syn[start:start + chunk_size]
        dists, _ = nn2.kneighbors(chunk)  # [chunk, 2]: (nearest, 2nd-nearest) real distances
        d1 = dists[:, 0]
        d2 = np.where(dists[:, 1] > 0, dists[:, 1], 1e-12)
        dcr_all[start:start + chunk_size] = d1
        nndr_all[start:start + chunk_size] = d1 / d2

    return {
        "dcr_mean": float(np.mean(dcr_all)),
        "dcr_median": float(np.median(dcr_all)),
        "dcr_min": float(np.min(dcr_all)),
        "nndr_mean": float(np.mean(nndr_all)),
        "nndr_median": float(np.median(nndr_all)),
        "exact_copy_ratio": float(np.mean(dcr_all < exact_copy_eps)),
        "near_copy_ratio": float(np.mean(dcr_all < near_copy_threshold)),
        "near_copy_threshold": near_copy_threshold,
    }


# ======================================================================
# Aggregation helper
# ======================================================================
def summarize_across_seeds(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """Mean +/- std across seeds, grouped by group_cols (e.g. ['configuration',
    'dataset', 'classifier'])."""
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()
