"""
preprocessing.py -- leak-free preprocessing pipeline (Sections 8-9), plus
the ORIGINAL-COLUMN feature-group bookkeeping the discriminator's feature
head now requires (Section 9 of the follow-up revision).

Strict protocol, per seed:
    1. Stratified 80:20 train/test split on RAW data.
    2. Fit OneHotEncoder ONLY on the training partition.
    3. Fit StandardScaler ONLY on the training partition.
    4. Transform train with the fitted preprocessor.
    5. Transform test with the SAME fitted preprocessor.
    6. CD-GAN is trained ONLY on the transformed training partition.
    7. Test-set information never touches GAN training or preprocessor fitting.

Rev 4 additions (reviewer point 1 -- preprocessing fully specified):
    * Missing values are handled EXPLICITLY and fit on TRAIN only:
      numerical -> train median; categorical -> a dedicated
      "__MISSING__" category (ARFF "?", empty strings, "nan"/"None"/"NA"
      are all treated as missing). Rev 3 had no imputation, so NaNs in
      numerical columns (e.g. SICK) passed through StandardScaler into the
      GAN.
    * Predictor columns that are ENTIRELY missing in the training
      partition (e.g. SICK's TBG) are dropped and listed in
      ``dropped_all_missing_cols``.
    * ``preprocessing_report`` writes a per-column table (type, missing
      counts in train/test, imputation value, #categories, unseen test
      categories, encoded width, scaler mean/SD) for the paper.
    * ``impute_dataframe`` returns the cleaned + imputed raw-space frame
      that the GAN actually sees, so fidelity is computed against it.

Encoded column order is fixed as:
    [ one-hot encoded categorical features | standardized numerical features ]

This ordering is what ``original_feature_groups`` / categorical-projection
logic below assumes, and is what the discriminator's feature head is built
from (never guessed inside the model).

Also computed here, from TRAIN ONLY, and carried on ``FittedPreprocessor``
for evaluation-time use (never used inside GAN training):
    * ``numerical_percentile_bounds`` -- (p1, p99) per numerical column, for
      the normalized-Wasserstein-distance metric in evaluation.py.
    * ``binary_category_orders`` -- canonical (level0, level1) order for
      every categorical column with exactly 2 categories, for a
      deterministic point-biserial sign convention in evaluation.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_CATEGORY = "__MISSING__"
MISSING_TOKENS = {"?", "", "nan", "NaN", "None", "none", "NA", "N/A", "null"}


def clean_categorical(series: pd.Series) -> pd.Series:
    """Decode bytes, strip whitespace, map every missing marker to the
    explicit MISSING_CATEGORY, and return strings."""
    def _c(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return MISSING_CATEGORY
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        v = str(v).strip()
        return MISSING_CATEGORY if v in MISSING_TOKENS else v
    return series.map(_c).astype(str)


def clean_numerical(series: pd.Series) -> pd.Series:
    """Coerce to float; ARFF '?' and any non-numeric token become NaN."""
    s = series.map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return pd.to_numeric(s.replace(list(MISSING_TOKENS), np.nan), errors="coerce").astype(np.float64)


@dataclass
class FittedPreprocessor:
    """Bundle of everything needed to encode/decode one seed's split.

    categorical_slices: list of (start, end) index ranges into the encoded
        [B, D] array, one per ORIGINAL categorical column, in the same
        order as ``categorical_cols``. Used both for one-hot projection
        of generated data and for feature-group-aware evaluation.
    numerical_slice: (start, end) covering all standardized numerical
        columns as a single contiguous block (each numerical column is
        width 1, so this is just the tail of the encoded vector).
    original_feature_groups: list of (start, end, name, dtype) tuples, ONE
        PER ORIGINAL PREDICTOR COLUMN (categorical AND numerical), in
        encoded-column order. ``dtype`` is "categorical" or "numerical".
        For a categorical column this is its COMPLETE one-hot block
        (width = its category count); for a numerical column this is its
        single standardized dimension. This is the exact mapping passed
        to ``DualHeadDiscriminator`` -- the model never guesses it.
    numerical_percentile_bounds: {numerical_col: (p1, p99)}, computed on
        RAW TRAINING values only. Used only for the evaluation-time
        normalized-Wasserstein-distance metric, never inside training.
    binary_category_orders: {categorical_col: [level0, level1]} for every
        categorical column with exactly 2 categories, in the SAME order
        ``OneHotEncoder`` assigned them (sklearn sorts categories
        lexicographically by default). Used only for a deterministic
        point-biserial sign convention in evaluation.py.
    """

    ohe: Optional[OneHotEncoder]
    scaler: Optional[StandardScaler]
    categorical_cols: List[str]
    numerical_cols: List[str]
    categorical_slices: List[Tuple[int, int]]
    categorical_categories: List[List] = field(default_factory=list)  # per-column category labels
    numerical_slice: Tuple[int, int] = (0, 0)
    original_feature_groups: List[Tuple[int, int, str, str]] = field(default_factory=list)
    numerical_percentile_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    binary_category_orders: Dict[str, List] = field(default_factory=dict)
    numerical_impute_values: Dict[str, float] = field(default_factory=dict)
    dropped_all_missing_cols: List[str] = field(default_factory=list)
    missing_counts_train: Dict[str, int] = field(default_factory=dict)
    encoded_dim: int = 0
    target_column: str = "target"
    target_mapping: Optional[Dict] = None
    seed: int = 0
    dataset_name: str = "dataset"

    @property
    def feature_group_slices(self) -> List[Tuple[int, int]]:
        """Just the (start, end) pairs from ``original_feature_groups``, in
        the exact form ``DualHeadDiscriminator`` expects."""
        return [(s, e) for (s, e, _name, _dtype) in self.original_feature_groups]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FittedPreprocessor":
        return joblib.load(path)


def validate_binary_target(y: np.ndarray, context_name: str) -> None:
    """Section 8 of the follow-up revision: STRICT binary-target validation.

    Raises if the (already target_mapping-applied) label array is not a
    subset of {0,1}, or does not contain BOTH classes. Called after every
    raw-label -> {0,1} mapping step (train fit, test/validation transform,
    and dataset-level diagnostics) so a mis-specified ``target_mapping``
    (wrong direction, a missed category, a typo) fails loudly at the
    earliest possible point rather than silently producing a degenerate
    single-class training set.
    """
    y_int = np.asarray(y).astype(int)
    uniq = set(np.unique(y_int).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"[{context_name}] target values are not a subset of {{0,1}} after mapping: {sorted(uniq)}"
        )
    if uniq != {0, 1}:
        raise ValueError(
            f"[{context_name}] target after mapping does not contain BOTH classes 0 and 1: found {sorted(uniq)}"
        )


def stratified_raw_split(
    df: pd.DataFrame,
    target_column: str,
    seed: int,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split RAW (untransformed) rows, stratified on the raw target column."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=int(seed),
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_preprocessor(
    train_df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_column: str,
    seed: int,
    dataset_name: str,
    target_mapping: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, FittedPreprocessor]:
    """Fit OneHotEncoder + StandardScaler on TRAIN ONLY.

    Returns (X_train_encoded [N,D], y_train [N], fitted preprocessor).
    """
    y_train_raw = train_df[target_column]
    if target_mapping is not None:
        y_train = y_train_raw.map(target_mapping).to_numpy()
    else:
        y_train = y_train_raw.to_numpy()
    if np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(y_train.dtype, np.integer):
        y_train = y_train.astype(np.float32)
    else:
        raise ValueError(
            "target_column could not be resolved to a numeric 0/1 label; "
            "pass target_mapping={'neg_label': 0, 'pos_label': 1}."
        )
    validate_binary_target(y_train, f"{dataset_name} (TRAIN fit, seed={seed})")

    # ---- Rev 4: explicit, train-only missing-value handling ----
    cat_frame = pd.DataFrame({c: clean_categorical(train_df[c]) for c in categorical_cols}, index=train_df.index)
    num_frame = pd.DataFrame({c: clean_numerical(train_df[c]) for c in numerical_cols}, index=train_df.index)
    missing_counts_train: Dict[str, int] = {}
    for c in categorical_cols:
        missing_counts_train[c] = int((cat_frame[c] == MISSING_CATEGORY).sum())
    for c in numerical_cols:
        missing_counts_train[c] = int(num_frame[c].isna().sum())
    n_rows = len(train_df)
    dropped_all_missing_cols = [c for c in list(categorical_cols) + list(numerical_cols)
                                if n_rows > 0 and missing_counts_train[c] == n_rows]
    if dropped_all_missing_cols:
        print(f"[{dataset_name} seed={seed}] dropping predictor columns that are entirely missing in TRAIN: "
              f"{dropped_all_missing_cols}")
    categorical_cols = [c for c in categorical_cols if c not in dropped_all_missing_cols]
    numerical_cols = [c for c in numerical_cols if c not in dropped_all_missing_cols]
    numerical_impute_values = {c: float(num_frame[c].median()) for c in numerical_cols}
    for c in numerical_cols:
        num_frame[c] = num_frame[c].fillna(numerical_impute_values[c])

    cat_blocks = []
    ohe = None
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoded = ohe.fit_transform(cat_frame[categorical_cols])
        # bookkeeping of per-original-column slices within the OHE block
        start = 0
        for col_categories in ohe.categories_:
            width = len(col_categories)
            cat_blocks.append((start, start + width))
            start += width
    else:
        cat_encoded = np.zeros((len(train_df), 0), dtype=np.float32)

    scaler = None
    if numerical_cols:
        scaler = StandardScaler()
        num_encoded = scaler.fit_transform(num_frame[numerical_cols].to_numpy(dtype=np.float64)).astype(np.float32)
    else:
        num_encoded = np.zeros((len(train_df), 0), dtype=np.float32)

    X_train = np.concatenate([cat_encoded.astype(np.float32), num_encoded], axis=1)

    n_cat_dims = cat_encoded.shape[1]
    numerical_slice = (n_cat_dims, n_cat_dims + num_encoded.shape[1])

    # ---- Section 9: ORIGINAL-COLUMN feature groups (categorical block-wise,
    # numerical single-dim), built from TRAIN-fitted OHE/scaler only. This
    # is the exact mapping handed to DualHeadDiscriminator -- never guessed
    # by the model. ----
    original_feature_groups: List[Tuple[int, int, str, str]] = []
    for col, (start, end) in zip(categorical_cols, cat_blocks):
        original_feature_groups.append((start, end, col, "categorical"))
    for i, col in enumerate(numerical_cols):
        start = numerical_slice[0] + i
        original_feature_groups.append((start, start + 1, col, "numerical"))

    # ---- Section 15: robust normalization bounds for numerical columns,
    # derived from RAW TRAINING values only (1st/99th percentile). ----
    numerical_percentile_bounds: Dict[str, Tuple[float, float]] = {}
    for col in numerical_cols:
        vals = num_frame[col].to_numpy(dtype=np.float64)
        if vals.size == 0:
            numerical_percentile_bounds[col] = (0.0, 1.0)
        else:
            p1, p99 = np.percentile(vals, [1, 99])
            if p99 <= p1:
                p99 = p1 + 1e-6
            numerical_percentile_bounds[col] = (float(p1), float(p99))

    # ---- Section 17: canonical binary-category order for a deterministic
    # point-biserial sign convention (sklearn's OneHotEncoder already sorts
    # categories lexicographically, so this is just surfacing that order). ----
    binary_category_orders: Dict[str, List] = {}
    if ohe is not None:
        for col, cats in zip(categorical_cols, ohe.categories_):
            if len(cats) == 2:
                binary_category_orders[col] = list(cats)

    fp = FittedPreprocessor(
        ohe=ohe,
        scaler=scaler,
        categorical_cols=list(categorical_cols),
        numerical_cols=list(numerical_cols),
        categorical_slices=cat_blocks,
        categorical_categories=[list(c) for c in (ohe.categories_ if ohe is not None else [])],
        numerical_slice=numerical_slice,
        original_feature_groups=original_feature_groups,
        numerical_percentile_bounds=numerical_percentile_bounds,
        binary_category_orders=binary_category_orders,
        numerical_impute_values=numerical_impute_values,
        dropped_all_missing_cols=dropped_all_missing_cols,
        missing_counts_train=missing_counts_train,
        encoded_dim=X_train.shape[1],
        target_column=target_column,
        target_mapping=target_mapping,
        seed=int(seed),
        dataset_name=dataset_name,
    )
    return X_train, y_train, fp


def impute_dataframe(df: pd.DataFrame, fp: "FittedPreprocessor") -> pd.DataFrame:
    """Rev 4: the cleaned + TRAIN-imputed raw-space predictors (only the
    columns the preprocessor kept), i.e. exactly what the GAN is trained
    on before encoding. Fidelity metrics compare synthetic data with this
    frame, so imputation is applied identically to every method."""
    out = {}
    for c in fp.categorical_cols:
        out[c] = clean_categorical(df[c])
    for c in fp.numerical_cols:
        out[c] = clean_numerical(df[c]).fillna(fp.numerical_impute_values.get(c, 0.0))
    return pd.DataFrame(out, index=df.index).reset_index(drop=True)


def preprocessing_report(fp: "FittedPreprocessor", train_df: pd.DataFrame, test_df: pd.DataFrame,
                         test_size: float) -> Dict[str, object]:
    """Rev 4 (reviewer point 1): everything the paper's preprocessing
    subsection needs, computed from the fitted (train-only) preprocessor."""
    rows = []
    for (start, end, name, dtype) in fp.original_feature_groups:
        rec = {"column": name, "type": dtype, "encoded_start": start, "encoded_width": end - start,
               "n_missing_train": fp.missing_counts_train.get(name, 0)}
        if dtype == "categorical":
            cats = fp.categorical_categories[fp.categorical_cols.index(name)]
            test_clean = clean_categorical(test_df[name])
            unseen = ~test_clean.isin([str(c) for c in cats])
            rec.update({
                "encoding": "one-hot (train categories; unseen -> all-zero block)",
                "imputation": f"explicit category '{MISSING_CATEGORY}'",
                "n_categories_train": len(cats),
                "has_missing_category": MISSING_CATEGORY in [str(c) for c in cats],
                "n_missing_test": int((test_clean == MISSING_CATEGORY).sum()),
                "n_unseen_test_categories": int(test_clean[unseen].nunique()),
                "n_test_rows_with_unseen_category": int(unseen.sum()),
            })
        else:
            i = fp.numerical_cols.index(name)
            rec.update({
                "encoding": "z-score (StandardScaler fit on train)",
                "imputation": f"train median = {fp.numerical_impute_values[name]:.6g}",
                "n_missing_test": int(clean_numerical(test_df[name]).isna().sum()),
                "scaler_mean": float(fp.scaler.mean_[i]),
                "scaler_sd": float(fp.scaler.scale_[i]),
            })
        rows.append(rec)
    return {
        "dataset": fp.dataset_name, "seed": fp.seed,
        "split": {"type": "stratified on target, raw data, before any fitting", "test_size": test_size,
                  "n_train": int(len(train_df)), "n_test": int(len(test_df))},
        "target": {"column": fp.target_column, "mapping": fp.target_mapping,
                   "note": "binary target mapped to {0,1}; used as the scalar generator/critic condition c"},
        "fit_scope": "imputation, one-hot categories and scaler statistics are fit on the training partition only",
        "encoded_layout": "[one-hot categorical blocks | standardized numerical columns]",
        "encoded_dim": fp.encoded_dim,
        "dropped_all_missing_cols": fp.dropped_all_missing_cols,
        "ordinal_treatment": "none: every categorical column (including ordered ones) is one-hot encoded",
        "generator_output": "linear output layer; each categorical block is converted to a valid one-hot vector "
                            "by argmax after generation, then decoded with the train-fitted encoder/scaler",
        "covariance_loss_space": "encoded, standardized space of the minibatch: real rows vs raw (pre-argmax) "
                                 "generator output",
        "columns": rows,
    }


def transform_with_preprocessor(
    df: pd.DataFrame,
    fp: FittedPreprocessor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform (e.g. test) data using an ALREADY-FITTED preprocessor.

    Never calls .fit / .fit_transform -- only .transform.
    """
    y_raw = df[fp.target_column]
    y = (y_raw.map(fp.target_mapping) if fp.target_mapping is not None else y_raw).to_numpy().astype(np.float32)
    validate_binary_target(y, f"{fp.dataset_name} (transform, seed={fp.seed})")

    clean = impute_dataframe(df, fp)
    if fp.categorical_cols:
        cat_encoded = fp.ohe.transform(clean[fp.categorical_cols]).astype(np.float32)
    else:
        cat_encoded = np.zeros((len(df), 0), dtype=np.float32)

    if fp.numerical_cols:
        num_encoded = fp.scaler.transform(clean[fp.numerical_cols].to_numpy(dtype=np.float64)).astype(np.float32)
    else:
        num_encoded = np.zeros((len(df), 0), dtype=np.float32)

    X = np.concatenate([cat_encoded, num_encoded], axis=1)
    return X, y


def prepare_seed_split(
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    target_mapping: Optional[Dict] = None,
) -> Dict[str, object]:
    """End-to-end leak-free split + fit + transform for ONE seed.

    Returns a dict with X_train, y_train, X_test, y_test, and the fitted
    preprocessor (which the caller should persist with
    ``fp.save(...)`` for later inverse-transform / generation).
    """
    train_raw, test_raw = train_test_split(
        df, test_size=test_size, stratify=df[target_column], random_state=int(seed), shuffle=True,
    )
    # Rev 4: positions of the train/test rows in the loaded dataframe, so
    # run_utils.split_fingerprint can prove that CD-GAN and every baseline
    # used exactly the same rows (independent of pandas version/formatting).
    train_row_ids = df.index.get_indexer(train_raw.index)
    test_row_ids = df.index.get_indexer(test_raw.index)
    train_df, test_df = train_raw.reset_index(drop=True), test_raw.reset_index(drop=True)
    X_train, y_train, fp = fit_preprocessor(
        train_df, categorical_cols, numerical_cols, target_column, seed, dataset_name, target_mapping
    )
    X_test, y_test = transform_with_preprocessor(test_df, fp)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": fp,
        "train_df": train_df,
        "test_df": test_df,
        "train_row_ids": train_row_ids,
        "test_row_ids": test_row_ids,
    }


def prepare_tuning_split(
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    target_mapping: Optional[Dict] = None,
) -> Dict[str, object]:
    """Leak-free THREE-WAY split for legitimate hyperparameter tuning /
    sensitivity analysis / performance-regression checking (Section 15 of
    the follow-up revision -- "never use test data to optimize").

    Protocol:
        1. Stratified raw split into DEV (1 - test_size, default 80%) and
           TEST (test_size, default 20%). TEST is set aside here and never
           touched again by this function or its caller -- it exists only
           so ``X_test``/``y_test`` are available for a FINAL, one-shot
           evaluation after all tuning is complete; no code path in
           ``sensitivity.py`` / ``regression_check.py`` may use it for
           selection.
        2. DEV is itself further stratified-split into an internal
           GAN-TRAIN partition ((1 - validation_size) of DEV) and an
           internal VALIDATION partition (validation_size of DEV, e.g.
           0.2 * 0.8 = 16% of the original data with the defaults below).
        3. Preprocessing (OneHotEncoder + StandardScaler) is fit ONLY on
           the internal GAN-TRAIN partition -- never on the full DEV
           partition, and never anywhere near TEST.

    Returns a dict with X_gan_train/y_gan_train, X_validation/y_validation
    (both encoded with the GAN-TRAIN-fitted preprocessor), X_test/y_test
    (transformed with that SAME preprocessor, held out for the caller's
    final evaluation only), the fitted preprocessor, and the raw
    dataframes for each partition.
    """
    dev_df, test_df = stratified_raw_split(df, target_column, seed=seed, test_size=test_size)
    gan_train_df, validation_df = stratified_raw_split(dev_df, target_column, seed=seed, test_size=validation_size)

    X_gan_train, y_gan_train, fp = fit_preprocessor(
        gan_train_df, categorical_cols, numerical_cols, target_column, seed, dataset_name, target_mapping
    )
    X_validation, y_validation = transform_with_preprocessor(validation_df, fp)
    X_test, y_test = transform_with_preprocessor(test_df, fp)

    return {
        "X_gan_train": X_gan_train,
        "y_gan_train": y_gan_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": fp,
        "dev_df": dev_df,
        "gan_train_df": gan_train_df,
        "validation_df": validation_df,
        "test_df": test_df,
    }


# ======================================================================
# Section 9: projecting generated categorical blocks to a valid one-hot
# ======================================================================
def project_categorical_blocks(
    X: np.ndarray,
    categorical_slices: List[Tuple[int, int]],
) -> np.ndarray:
    """Hard-project every categorical OHE block onto a valid one-hot vector.

    For each original categorical variable's block [start, end):
        1. find the argmax entry,
        2. set it to 1,
        3. set every other entry in that block to 0.

    Numerical (non-block) entries are left untouched (they stay continuous).
    Accepts and returns a numpy array; a torch-tensor variant is provided
    in ``project_categorical_blocks_torch`` for use inside the training
    loop / generation code without a CPU round-trip.
    """
    X_proj = X.copy()
    for start, end in categorical_slices:
        if end <= start:
            continue
        block = X[:, start:end]
        argmax_idx = np.argmax(block, axis=1)
        one_hot = np.zeros_like(block)
        one_hot[np.arange(block.shape[0]), argmax_idx] = 1.0
        X_proj[:, start:end] = one_hot
    return X_proj


def project_categorical_blocks_torch(X, categorical_slices: List[Tuple[int, int]]):
    """Torch-tensor version of ``project_categorical_blocks`` (no grad needed;
    this is a post-hoc, non-differentiable hard projection used only for
    inverse-transform / evaluation / downstream classifier input, never
    inside the differentiable generator forward pass)."""
    import torch

    X_proj = X.clone()
    for start, end in categorical_slices:
        if end <= start:
            continue
        block = X[:, start:end]
        argmax_idx = torch.argmax(block, dim=1)
        one_hot = torch.zeros_like(block)
        one_hot.scatter_(1, argmax_idx.unsqueeze(1), 1.0)
        X_proj[:, start:end] = one_hot
    return X_proj


def inverse_transform(
    X_encoded: np.ndarray,
    fp: FittedPreprocessor,
) -> pd.DataFrame:
    """Decode an encoded [N, D] array (already categorical-projected) back to
    a raw-space DataFrame using the SAME fitted preprocessor."""
    n_cat_dims = fp.categorical_slices[-1][1] if fp.categorical_slices else 0
    out = {}
    if fp.categorical_cols:
        cat_block = X_encoded[:, :n_cat_dims]
        cat_raw = fp.ohe.inverse_transform(cat_block)
        for i, col in enumerate(fp.categorical_cols):
            out[col] = cat_raw[:, i]
    if fp.numerical_cols:
        num_block = X_encoded[:, n_cat_dims:]
        num_raw = fp.scaler.inverse_transform(num_block)
        for i, col in enumerate(fp.numerical_cols):
            out[col] = num_raw[:, i]
    return pd.DataFrame(out)


def save_config_snapshot(fp: FittedPreprocessor, out_path: str) -> None:
    """Human-readable JSON summary of the fitted preprocessor (for configs/),
    including the ORIGINAL-COLUMN feature-group mapping handed to the
    discriminator (Section 9: "store this mapping in config/metadata")."""
    payload = {
        "dataset_name": fp.dataset_name,
        "seed": fp.seed,
        "target_column": fp.target_column,
        "target_mapping": fp.target_mapping,
        "categorical_cols": fp.categorical_cols,
        "numerical_cols": fp.numerical_cols,
        "categorical_slices": fp.categorical_slices,
        "numerical_slice": fp.numerical_slice,
        "encoded_dim": fp.encoded_dim,
        "original_feature_groups": [
            {"start": s, "end": e, "name": name, "dtype": dtype}
            for (s, e, name, dtype) in fp.original_feature_groups
        ],
        "n_original_feature_groups": len(fp.original_feature_groups),
        "numerical_percentile_bounds": fp.numerical_percentile_bounds,
        "binary_category_orders": fp.binary_category_orders,
        "numerical_impute_values": fp.numerical_impute_values,
        "dropped_all_missing_cols": fp.dropped_all_missing_cols,
        "missing_counts_train": fp.missing_counts_train,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
