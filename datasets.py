"""
datasets.py -- one generic dataset interface, used for every experiment.

The SAME CD-GAN implementation (model.py / trainer.py) is reused for every
dataset. A dataset only has to supply:

    dataframe          : pandas.DataFrame with raw (untransformed) values
    target_column      : str
    categorical_cols   : List[str]
    numerical_cols      : List[str]
    dataset_name       : str
    target_mapping     : Optional[Dict] mapping raw label values -> {0, 1}

No part of the GAN architecture is duplicated per dataset.

FINAL PRE-TRAINING REVISION -- SICK/NOMAO/MV/KDDCup09-Upselling loaders
--------------------------------------------------------------------------
NOMAO, MV, and KDDCup09-Upselling are now filled in with the EXACT column
definitions you supplied (verbatim), replacing the previous NotImplementedError
templates. SICK is unchanged from the earlier revision.

FLAGGED AMBIGUITY (do not silently resolve, per your instructions) -- NOMAO
target polarity: you specified ``target="class"`` "mapped explicitly to
{0,1}" for NOMAO, but -- unlike MV ("P"->1, "N"->0) and KDDCup09-Upselling
("-1"->0, "1"->1) -- you did not give the raw-label -> {0,1} DIRECTION for
NOMAO's two class values. ``load_nomao()`` below therefore:
  1. accepts an optional ``target_mapping`` argument so you can pass the
     correct direction explicitly, and
  2. if you do not, auto-detects the two raw label values, maps them by
     SORTED ORDER (raw_values[0] -> 0, raw_values[1] -> 1), and PRINTS a
     loud, explicit warning identifying exactly which raw value became 0
     and which became 1 -- so this is never silently guessed without you
     being told, and it is trivially overridable.

Every dataset's exact column split, target mapping, and (once fit) encoded
dimension / #original-feature-groups / class balance are printed via
``print_dataset_diagnostics`` (raw-space, called right after loading) and
in ``run_experiment.py`` (encoded-space, called right after the
preprocessor is fit) -- see Section 7 of your instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from preprocessing import validate_binary_target


@dataclass
class DatasetSpec:
    dataframe: pd.DataFrame
    target_column: str
    categorical_cols: List[str]
    numerical_cols: List[str]
    dataset_name: str
    target_mapping: Optional[Dict] = None

    def __post_init__(self):
        cols = set(self.dataframe.columns)
        if self.target_column not in cols:
            raise ValueError(f"[{self.dataset_name}] target_column '{self.target_column}' not found in dataframe.")

        overlap = set(self.categorical_cols) & set(self.numerical_cols)
        if overlap:
            raise ValueError(
                f"[{self.dataset_name}] columns listed as both categorical and numerical: {sorted(overlap)}"
            )

        declared = set(self.categorical_cols) | set(self.numerical_cols)
        missing_from_df = declared - cols
        if missing_from_df:
            raise ValueError(
                f"[{self.dataset_name}] categorical/numerical columns not found in dataframe: "
                f"{sorted(missing_from_df)}"
            )

        # ---- Section 7: STRICT schema validation -- the union of
        # categorical_cols + numerical_cols must equal EXACTLY the set of
        # predictor columns (every dataframe column other than the target),
        # with no omission and no unknown extra column. ----
        predictor_cols = cols - {self.target_column}
        omitted = predictor_cols - declared
        if omitted:
            raise ValueError(
                f"[{self.dataset_name}] columns present in the dataframe but NOT listed as categorical or "
                f"numerical (and not the target column) -- every predictor column must be explicitly "
                f"classified: {sorted(omitted)}"
            )
        extra = declared - predictor_cols
        if extra:
            raise ValueError(
                f"[{self.dataset_name}] columns listed as categorical/numerical that are not valid predictor "
                f"columns (e.g. equal to target_column, or otherwise not a real predictor): {sorted(extra)}"
            )


def print_dataset_diagnostics(spec: "DatasetSpec") -> None:
    """Section 7: raw-space diagnostics, printed right after loading (before
    any preprocessing/splitting). Encoded-space diagnostics (encoded_dim,
    #original_feature_groups) are printed separately in run_experiment.py
    once the preprocessor has actually been fit for a given seed, since
    those quantities depend on which categories are observed in that
    seed's TRAINING partition and cannot be known from the raw dataframe
    alone.
    """
    df = spec.dataframe
    n_predictors = len(spec.categorical_cols) + len(spec.numerical_cols)
    print(f"[{spec.dataset_name}] raw rows            : {len(df)}")
    print(
        f"[{spec.dataset_name}] #predictor columns  : {n_predictors} "
        f"({len(spec.categorical_cols)} categorical + {len(spec.numerical_cols)} numerical)"
    )

    if spec.target_mapping is not None:
        y_raw = df[spec.target_column]
        y_mapped = y_raw.map(spec.target_mapping)
        unmapped_mask = y_mapped.isna() & ~y_raw.isna()
        if unmapped_mask.any():
            raise ValueError(
                f"[{spec.dataset_name}] target_mapping {spec.target_mapping} does not cover all raw target "
                f"values; unmapped raw values: {sorted(y_raw[unmapped_mask].unique().tolist())}"
            )
        y_for_counts = y_mapped.dropna().astype(int).to_numpy()
    else:
        y_for_counts = df[spec.target_column].to_numpy()

    validate_binary_target(y_for_counts, spec.dataset_name)
    classes, counts = np.unique(y_for_counts, return_counts=True)
    total = counts.sum()
    for c, cnt in zip(classes, counts):
        print(f"[{spec.dataset_name}] class {int(c)}: {int(cnt)} rows ({100.0 * cnt / total:.2f}%)")


# ----------------------------------------------------------------------
# ARFF loading helper (shared by SICK, NOMAO, MV -- all distributed as
# ARFF). Decodes byte/string nominal columns back to plain str.
# ----------------------------------------------------------------------
def _load_arff_dataframe(arff_path: str) -> pd.DataFrame:
    from scipy.io import arff

    data, _meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return df


def _resolve_column_name(df: pd.DataFrame, name: str) -> str:
    """Exact match first; falls back to a UNIQUE case-insensitive match.
    Raises rather than guessing if the column cannot be found or the
    case-insensitive match is ambiguous."""
    if name in df.columns:
        return name
    matches = [c for c in df.columns if c.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Could not uniquely resolve column '{name}' among dataframe columns {list(df.columns)} "
        f"(case-insensitive matches: {matches})."
    )


def _normalize_pm1_target(series: pd.Series) -> pd.Series:
    """Normalizes a target column that may arrive as strings ('-1','1'),
    byte-strings, or numeric values (-1, 1, -1.0, 1.0) into the canonical
    string labels '-1'/'1', so a single {'-1': 0, '1': 1} target_mapping
    works regardless of the raw dtype (KDDCup09-Upselling's ``UPSELLING``
    column has been observed in both numeric and string raw forms)."""

    def _norm(v):
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        if isinstance(v, str):
            v = v.strip()
            try:
                return str(int(float(v)))
            except ValueError:
                return v
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return str(v)

    return series.apply(_norm)


# ----------------------------------------------------------------------
# SICK (schema confirmed from your existing loading code in this folder)
# ----------------------------------------------------------------------
def load_sick(arff_path: str) -> DatasetSpec:
    """dataset_38_sick.arff, target column 'Class' (binary: sick / negative)."""
    df = _load_arff_dataframe(arff_path)

    target_column = "Class"
    numerical_cols = [
        c for c in ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"] if c in df.columns
    ]
    categorical_cols = [
        c for c in df.columns if c not in numerical_cols + [target_column]
    ]
    target_mapping = {"negative": 0, "sick": 1}

    spec = DatasetSpec(
        dataframe=df,
        target_column=target_column,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dataset_name="SICK",
        target_mapping=target_mapping,
    )
    print_dataset_diagnostics(spec)
    return spec


# ----------------------------------------------------------------------
# NOMAO -- exact categorical column list as supplied. See the FLAGGED
# AMBIGUITY note in the module docstring re: target-label polarity.
# ----------------------------------------------------------------------
NOMAO_CATEGORICAL_COLS: List[str] = [
    "V7", "V8", "V15", "V16", "V23", "V24", "V31", "V32", "V39", "V40",
    "V47", "V48", "V55", "V56", "V63", "V64", "V71", "V72", "V79", "V80",
    "V87", "V88", "V92", "V96", "V100", "V104", "V108", "V112", "V116",
]


def load_nomao(arff_path: str, target_mapping: Optional[Dict] = None) -> DatasetSpec:
    """NOMAO: ARFF, "?" -> NaN with missing rows dropped, target column
    'class' mapped explicitly to {0,1} (see FLAGGED AMBIGUITY above re:
    which raw label maps to which of 0/1 -- pass ``target_mapping``
    explicitly to control this; otherwise a sorted-order default is used
    and loudly printed)."""
    df = _load_arff_dataframe(arff_path)
    target_column = _resolve_column_name(df, "class")

    before = len(df)
    df = df.replace("?", np.nan)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[NOMAO] dropped {dropped} of {before} raw rows containing '?' (missing values).")

    categorical_cols = [c for c in NOMAO_CATEGORICAL_COLS if c in df.columns]
    missing_expected = set(NOMAO_CATEGORICAL_COLS) - set(categorical_cols)
    if missing_expected:
        raise ValueError(
            f"[NOMAO] expected categorical columns not found in the ARFF file: {sorted(missing_expected)}. "
            "Refusing to silently drop them from the categorical list -- verify the file matches the "
            "column definitions supplied for this dataset."
        )
    numerical_cols = [c for c in df.columns if c not in categorical_cols + [target_column]]

    if target_mapping is None:
        raw_values = sorted(df[target_column].astype(str).unique().tolist())
        if len(raw_values) != 2:
            raise ValueError(f"[NOMAO] expected exactly 2 raw target values in column '{target_column}', found {raw_values}")
        target_mapping = {raw_values[0]: 0, raw_values[1]: 1}
        print(
            f"[NOMAO] FLAGGED AMBIGUITY: no explicit target_mapping was supplied and the raw-label->0/1 "
            f"direction was not specified in your instructions (unlike MV's P/N or KDDCup09's -1/1). "
            f"Defaulting to SORTED-ORDER mapping: {target_mapping}. VERIFY this matches your intended "
            f"polarity before trusting downstream TSTR/utility results -- pass target_mapping=... to "
            f"load_nomao() explicitly to override."
        )

    spec = DatasetSpec(
        dataframe=df,
        target_column=target_column,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dataset_name="NOMAO",
        target_mapping=target_mapping,
    )
    print_dataset_diagnostics(spec)
    return spec


# ----------------------------------------------------------------------
# MV -- exact categorical column list and target mapping as supplied.
# ----------------------------------------------------------------------
MV_CATEGORICAL_COLS: List[str] = ["x3", "x7", "x8"]


def load_mv(arff_path: str) -> DatasetSpec:
    """MV: ARFF, target column 'binaryClass', "P"->1, "N"->0."""
    df = _load_arff_dataframe(arff_path)
    target_column = _resolve_column_name(df, "binaryClass")

    categorical_cols = [c for c in MV_CATEGORICAL_COLS if c in df.columns]
    missing_expected = set(MV_CATEGORICAL_COLS) - set(categorical_cols)
    if missing_expected:
        raise ValueError(
            f"[MV] expected categorical columns not found in the ARFF file: {sorted(missing_expected)}. "
            "Refusing to silently drop them from the categorical list -- verify the file matches the "
            "column definitions supplied for this dataset."
        )
    numerical_cols = [c for c in df.columns if c not in categorical_cols + [target_column]]
    target_mapping = {"P": 1, "N": 0}

    spec = DatasetSpec(
        dataframe=df,
        target_column=target_column,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dataset_name="MV",
        target_mapping=target_mapping,
    )
    print_dataset_diagnostics(spec)
    return spec


# ----------------------------------------------------------------------
# KDDCup09 Upselling -- exact categorical column list and target mapping
# as supplied. Target handles BOTH numeric and string raw forms of -1/1.
# ----------------------------------------------------------------------
KDDCUP09_UPSELLING_CATEGORICAL_COLS: List[str] = [
    "Var194", "Var196", "Var201", "Var203", "Var205", "Var207", "Var208",
    "Var210", "Var211", "Var218", "Var221", "Var223", "Var225", "Var227", "Var229",
]


def load_kddcup09_upselling(csv_path: str) -> DatasetSpec:
    """KDDCup09 Upselling: target column 'UPSELLING', "-1"->0, "1"->1
    (handles both numeric and string raw target representations)."""
    df = pd.read_csv(csv_path)
    target_column = _resolve_column_name(df, "UPSELLING")
    df[target_column] = _normalize_pm1_target(df[target_column])

    categorical_cols = [c for c in KDDCUP09_UPSELLING_CATEGORICAL_COLS if c in df.columns]
    missing_expected = set(KDDCUP09_UPSELLING_CATEGORICAL_COLS) - set(categorical_cols)
    if missing_expected:
        raise ValueError(
            f"[KDDCUP09_UPSELLING] expected categorical columns not found in the CSV file: "
            f"{sorted(missing_expected)}. Refusing to silently drop them from the categorical list -- "
            "verify the file matches the column definitions supplied for this dataset."
        )
    numerical_cols = [c for c in df.columns if c not in categorical_cols + [target_column]]
    target_mapping = {"-1": 0, "1": 1}

    spec = DatasetSpec(
        dataframe=df,
        target_column=target_column,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dataset_name="KDDCUP09_UPSELLING",
        target_mapping=target_mapping,
    )
    print_dataset_diagnostics(spec)
    return spec


DATASET_LOADERS = {
    "SICK": load_sick,
    "NOMAO": load_nomao,
    "MV": load_mv,
    "KDDCUP09_UPSELLING": load_kddcup09_upselling,
}


def load_from_spec(spec_path: str) -> DatasetSpec:
    """Rev 4: generic loader for the remaining benchmark datasets. The JSON
    spec states every schema decision explicitly, so it can be reported:

        {
          "dataset_name": "ADULT",
          "path": "data/adult.csv",           # .csv or .arff; relative to the spec file
          "target_column": "class",
          "target_mapping": {"<=50K": 0, ">50K": 1},
          "categorical_cols": ["workclass", "education", ...],
          "numerical_cols": ["age", "fnlwgt", ...],
          "drop_cols": ["id"],                # optional
          "source": "OpenML ID 1590",         # optional, copied to diagnostics
          "notes": "binarized as ..."          # optional
        }

    ``categorical_cols``/``numerical_cols`` must cover every remaining
    predictor (DatasetSpec enforces this). Missing values are left as-is;
    preprocessing.py handles them with train-only imputation."""
    import json
    import os

    with open(spec_path) as f:
        cfg = json.load(f)
    path = cfg["path"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(spec_path)), path)
    if path.lower().endswith(".arff"):
        df = _load_arff_dataframe(path)
    else:
        df = pd.read_csv(path, na_values=cfg.get("na_values", ["?"]), keep_default_na=True)
    df = df.drop(columns=[c for c in cfg.get("drop_cols", []) if c in df.columns])
    target_column = _resolve_column_name(df, cfg["target_column"])
    mapping = cfg.get("target_mapping")
    if mapping is not None:
        # JSON keys are strings; compare against the target as strings
        df[target_column] = df[target_column].map(
            lambda v: v.decode("utf-8") if isinstance(v, bytes) else v).astype(str).str.strip()
        mapping = {str(k): int(v) for k, v in mapping.items()}
    spec = DatasetSpec(
        dataframe=df,
        target_column=target_column,
        categorical_cols=list(cfg["categorical_cols"]),
        numerical_cols=list(cfg["numerical_cols"]),
        dataset_name=cfg["dataset_name"],
        target_mapping=mapping,
    )
    if cfg.get("source"):
        print(f"[{spec.dataset_name}] source: {cfg['source']}")
    print_dataset_diagnostics(spec)
    return spec


def load_dataset(name: str, path: str) -> DatasetSpec:
    """Built-in loaders for SICK/NOMAO/MV/KDDCUP09_UPSELLING; any other
    dataset is loaded from a JSON spec (pass --dataset SPEC --data-path x.json,
    or any name with a .json path)."""
    name_u = name.upper()
    if name_u == "SPEC" or str(path).lower().endswith(".json"):
        return load_from_spec(path)
    if name_u not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(DATASET_LOADERS)} or SPEC (JSON file)")
    return DATASET_LOADERS[name_u](path)
