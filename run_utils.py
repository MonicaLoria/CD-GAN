"""
run_utils.py -- Rev 4: small helpers shared by run_experiment.py and
run_baselines.py so every run folder has the same bookkeeping files.
Torch is optional here (imported lazily).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from typing import Dict, Optional

import numpy as np
import pandas as pd


def write_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def split_fingerprint(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      train_row_ids: Optional[np.ndarray] = None,
                      test_row_ids: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Proof that two runs used the same real split. ``split_sha1`` hashes
    the sorted ROW POSITIONS of the train and test rows in the loaded
    dataset (robust to library versions); if positions are not supplied it
    falls back to a content hash. make_tables.py refuses to build tables
    when methods on the same (dataset, seed) disagree."""
    def _content(df: pd.DataFrame) -> str:
        rows = np.sort(pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy())
        return hashlib.sha1(rows.tobytes()).hexdigest()

    out: Dict[str, object] = {"n_train": int(len(train_df)), "n_test": int(len(test_df)),
                              "train_content_sha1": _content(train_df), "test_content_sha1": _content(test_df)}
    if train_row_ids is not None and test_row_ids is not None:
        tr = hashlib.sha1(np.sort(np.asarray(train_row_ids, dtype=np.int64)).tobytes()).hexdigest()
        te = hashlib.sha1(np.sort(np.asarray(test_row_ids, dtype=np.int64)).tobytes()).hexdigest()
        out.update({"basis": "row positions", "train_sha1": tr, "test_sha1": te})
    else:
        out.update({"basis": "row contents", "train_sha1": out["train_content_sha1"],
                    "test_sha1": out["test_content_sha1"]})
    out["split_sha1"] = hashlib.sha1((out["train_sha1"] + out["test_sha1"]).encode()).hexdigest()
    return out


def package_versions(extra: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    import scipy
    import sklearn
    v: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    for mod in ["torch", "imblearn", "sdv", "ctgan", "lightgbm"]:
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            v[mod] = None
    try:
        import torch
        v["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    if extra:
        v.update(extra)
    return v


def export_split(split: Dict, out_dir: str, target_column: str) -> None:
    """Writes the raw train/test partitions so external baselines (TWAE,
    TabDDPM, CTAB-GAN+, TABULA, ...) can be trained on exactly the same
    training rows and scored by quality_metrics.evaluate_synthetic."""
    os.makedirs(out_dir, exist_ok=True)
    split["train_df"].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    split["test_df"].to_csv(os.path.join(out_dir, "test.csv"), index=False)
    write_json({"target_column": target_column,
                **split_fingerprint(split["train_df"], split["test_df"],
                                    split.get("train_row_ids"), split.get("test_row_ids"))},
               os.path.join(out_dir, "split_fingerprint.json"))
