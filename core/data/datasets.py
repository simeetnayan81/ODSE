"""Dataset registry for the ODSE sandbox environment.

All datasets are generated in-process using ``sklearn.datasets`` and
plain ``numpy`` / ``pandas`` - **no network downloads required**.

Each dataset bundles a DataFrame with metadata (problem type, target column,
feature columns). Datasets are keyed by ``(name, difficulty)`` and loaded
lazily via factory functions.

Adding a new dataset
--------------------
1. Write a loader function that returns ``DatasetConfig``.
2. Add an entry to ``_REGISTRY`` at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)

from ..models import Difficulty, ProblemType

# ============================================================================
# DatasetConfig
# ============================================================================

class DatasetConfig:
    """Bundles a DataFrame with modelling metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    target_column : str
        Name of the target column.
    problem_type : ProblemType
        Classification or regression.
    feature_columns : List[str] | None
        Explicit feature list; if *None*, all non-target / non-excluded
        columns are used.
    exclude_columns : List[str] | None
        Columns to exclude from features (IDs, free text, ...).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        feature_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> None:
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.exclude_columns = exclude_columns or []
        self.feature_columns: List[str] = feature_columns or [
            c
            for c in df.columns
            if c != target_column and c not in self.exclude_columns
        ]

# ============================================================================
# Public API
# ============================================================================

def load_dataset(
    name: str,
    difficulty: Difficulty | str = Difficulty.EASY,
) -> DatasetConfig:
    """Load a dataset by *name* and *difficulty*.

    Falls back to a difficulty-agnostic entry if the exact key is missing.
    Raises ``ValueError`` when no match is found.
    """
    if isinstance(difficulty, str):
        difficulty = Difficulty(difficulty)

    key: _RegistryKey = (name, difficulty)
    loader = _REGISTRY.get(key)

    if loader is None:
        # Fall back to difficulty-agnostic entry
        loader = _REGISTRY.get((name, None))

    if loader is None:
        available = sorted([k[0] for k in _REGISTRY])
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return loader()

def list_datasets() -> List[Dict[str, str]]:
    """Return a summary of all registered datasets."""
    datasets: Dict[str, List[str]] = {}
    for name, diff in _REGISTRY:
        datasets.setdefault(name, [])
        if diff is not None:
            datasets[name].append(diff.value)
    return [
        {"name": n, "difficulties": sorted(d)} for n, d in datasets.items()
    ]

# ============================================================================
# Helpers
# ============================================================================

def _inject_nulls(
    df: pd.DataFrame,
    columns: List[str],
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Inject NaN into *columns* at the given *fraction*."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mask = rng.rand(len(df)) < fraction
            df.loc[mask, col] = np.nan
    return df

def _add_categorical_column(
    df: pd.DataFrame,
    col_name: str,
    categories: List[str],
    seed: int,
) -> pd.DataFrame:
    """Add a random categorical column to *df*."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[col_name] = rng.choice(categories, size=len(df))
    return df

# ============================================================================
# Dataset Loaders - all offline, no network required
# ============================================================================

# -- Breast Cancer (binary classification) -----------------------------------

def _load_breast_cancer_easy() -> DatasetConfig:
    """Breast cancer - binary classification, clean, 30 numeric features."""
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.CLASSIFICATION,
    )

def _load_breast_cancer_medium() -> DatasetConfig:
    """Breast cancer with ~15 % nulls injected."""
    cfg = _load_breast_cancer_easy()
    df = _inject_nulls(
        cfg.df,
        columns=["mean radius", "mean texture", "mean perimeter", "mean area"],
        fraction=0.15,
        seed=123,
    )
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.CLASSIFICATION,
    )

def _load_breast_cancer_hard() -> DatasetConfig:
    """Breast cancer with ~25 % nulls + noise columns."""
    cfg = _load_breast_cancer_easy()
    rng = np.random.RandomState(456)
    df = _inject_nulls(
        cfg.df,
        columns=[c for c in cfg.df.columns if c != "target"],
        fraction=0.25,
        seed=456,
    )
    df["noise_a"] = rng.randn(len(df))
    df["noise_b"] = rng.choice(["x", "y", "z"], size=len(df))
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.CLASSIFICATION,
    )

# -- Iris (multi-class classification) ---------------------------------------

def _load_iris_easy() -> DatasetConfig:
    """Iris - 3-class classification, 4 clean numeric features."""
    bunch = load_iris()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["species"] = pd.Categorical.from_codes(bunch.target, bunch.target_names)
    return DatasetConfig(
        df=df,
        target_column="species",
        problem_type=ProblemType.CLASSIFICATION,
    )

# -- Wine (multi-class classification) ---------------------------------------

def _load_wine_easy() -> DatasetConfig:
    """Wine - 3-class classification, 13 numeric features."""
    bunch = load_wine()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["quality_class"] = bunch.target
    return DatasetConfig(
        df=df,
        target_column="quality_class",
        problem_type=ProblemType.CLASSIFICATION,
    )

def _load_wine_medium() -> DatasetConfig:
    """Wine with nulls + a categorical column."""
    cfg = _load_wine_easy()
    df = _inject_nulls(cfg.df, columns=["alcohol", "ash", "magnesium"], fraction=0.20, seed=321)
    df = _add_categorical_column(df, "region", ["north", "south", "east", "west"], seed=321)
    return DatasetConfig(
        df=df,
        target_column="quality_class",
        problem_type=ProblemType.CLASSIFICATION,
    )

# -- Synthetic classification (scalable) -------------------------------------

def _load_synth_cls_easy() -> DatasetConfig:
    """Synthetic binary classification - 10 features, 500 samples, clean."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.CLASSIFICATION,
    )

def _load_synth_cls_hard() -> DatasetConfig:
    """Synthetic multi-class - 20 features, 1000 samples, nulls + noise."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=4, n_classes=4, n_clusters_per_class=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    df = _inject_nulls(df, columns=["f0", "f3", "f7", "f12"], fraction=0.15, seed=99)
    df = _add_categorical_column(df, "group", ["A", "B", "C"], seed=99)
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.CLASSIFICATION,
    )

# -- Regression (make_regression based) --------------------------------------

def _load_regression_easy() -> DatasetConfig:
    """Simple regression - 8 features, 400 samples, clean."""
    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=5,
        noise=10.0, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
    )

def _load_regression_medium() -> DatasetConfig:
    """Medium regression - 12 features, 600 samples, some nulls."""
    X, y = make_regression(
        n_samples=600, n_features=12, n_informative=7,
        noise=15.0, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    df = _inject_nulls(df, columns=["f1", "f4", "f8"], fraction=0.10, seed=55)
    df = _add_categorical_column(df, "category", ["low", "mid", "high"], seed=55)
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
    )

def _load_regression_hard() -> DatasetConfig:
    """Hard regression - 20 features, 1000 samples, heavy nulls + noise cols."""
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=10,
        noise=25.0, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    df = _inject_nulls(
        df, columns=[f"f{i}" for i in range(0, 20, 3)], fraction=0.20, seed=77
    )

    rng = np.random.RandomState(77)
    df["noise_a"] = rng.randn(len(df))
    df["noise_b"] = rng.choice(["x", "y", "z"], size=len(df))
    df = _add_categorical_column(df, "region", ["north", "south", "east", "west"], seed=77)
    return DatasetConfig(
        df=df,
        target_column="target",
        problem_type=ProblemType.REGRESSION,
    )

# -- House price (synthetic, realistic column names) -------------------------

def _load_house_price() -> DatasetConfig:
    """Synthetic house-price dataset with realistic column names."""
    rng = np.random.RandomState(42)
    n = 600
    sqft = rng.normal(1800, 400, n).clip(600, 5000)
    bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.20, 0.40, 0.25, 0.10])
    bathrooms = rng.choice([1, 2, 3], n, p=[0.25, 0.50, 0.25])
    age = rng.randint(0, 80, n)
    garage = rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
    neighborhood = rng.choice(["downtown", "suburb", "rural"], n, p=[0.3, 0.5, 0.2])

    price = (
        50_000
        + 120 * sqft
        + 15_000 * bedrooms
        + 12_000 * bathrooms
        - 800 * age
        + 20_000 * garage
        + rng.normal(0, 25_000, n)
    )

    df = pd.DataFrame({
        "sqft": sqft.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "garage": garage,
        "neighborhood": neighborhood,
        "price": price.round(0).astype(int),
    })
    return DatasetConfig(
        df=df,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
    )

# ============================================================================
# Registry - (name, Difficulty | None) -> loader callable
# ============================================================================

_RegistryKey = Tuple[str, Optional[Difficulty]]

_REGISTRY: Dict[_RegistryKey, Callable[[], DatasetConfig]] = {
    # -- Classification ------------------------------------------------------
    ("breast_cancer", Difficulty.EASY): _load_breast_cancer_easy,
    ("breast_cancer", Difficulty.MEDIUM): _load_breast_cancer_medium,
    ("breast_cancer", Difficulty.HARD): _load_breast_cancer_hard,
    ("breast_cancer", None): _load_breast_cancer_easy,
    ("iris", Difficulty.EASY): _load_iris_easy,
    ("iris", None): _load_iris_easy,
    ("wine", Difficulty.EASY): _load_wine_easy,
    ("wine", Difficulty.MEDIUM): _load_wine_medium,
    ("wine", None): _load_wine_easy,
    ("synth_cls", Difficulty.EASY): _load_synth_cls_easy,
    ("synth_cls", Difficulty.HARD): _load_synth_cls_hard,
    ("synth_cls", None): _load_synth_cls_easy,
    # -- Regression ----------------------------------------------------------
    ("regression", Difficulty.EASY): _load_regression_easy,
    ("regression", Difficulty.MEDIUM): _load_regression_medium,
    ("regression", Difficulty.HARD): _load_regression_hard,
    ("regression", None): _load_regression_easy,
    ("house_price", Difficulty.EASY): _load_house_price,
    ("house_price", None): _load_house_price,
}