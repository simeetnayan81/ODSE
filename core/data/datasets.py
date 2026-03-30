"""Dataset registry provides datasets for each (TaskType, Difficulty) pair.

Each loader returns a ``DatasetConfig`` that bundles a DataFrame with its
target column, feature columns, and any columns to exclude from modelling.

To add a new dataset:
1. Write a loader function that returns ``DatasetConfig``.
2. Add an entry to ``_REGISTRY`` at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns  # Used to load standard datasets as pandas DataFrames

from ..models import Difficulty, TaskType

# -----------------------------------------------------------------------------
# DatasetConfig
# -----------------------------------------------------------------------------

class DatasetConfig:
    """Bundles a DataFrame with modelling metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    target_column : str
        Name of the target (label) column.
    feature_columns : list[str] | None
        Explicit feature list. If *None*, all columns except
        *target_column* and *exclude_columns* are used.
    exclude_columns : list[str] | None
        Columns to exclude from features (e.g. IDs, free-text).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> None:
        self.df = df
        self.target_column = target_column
        self.exclude_columns = exclude_columns or []
        self.feature_columns: List[str] = feature_columns or [
            c
            for c in df.columns
            if c != target_column and c not in self.exclude_columns
        ]

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_dataset(task_type: TaskType, difficulty: Difficulty) -> DatasetConfig:
    """Load the appropriate dataset for a given task and difficulty.

    Raises ``ValueError`` if no dataset is registered for the pair.
    """
    loader = _REGISTRY.get((task_type, difficulty))
    if loader is None:
        raise ValueError(
            f"No dataset registered for task={task_type.value}, "
            f"difficulty={difficulty.value}"
        )
    return loader()

# -----------------------------------------------------------------------------
# Dataset loaders (private)
# -----------------------------------------------------------------------------

def _load_titanic_easy() -> DatasetConfig:
    """Titanic dataset with moderate nulls - easy cleaning."""
    df = sns.load_dataset("titanic")
    return DatasetConfig(
        df=df,
        target_column="survived",
        # Exclude text/leaky columns to keep the easy task simple
        exclude_columns=["alive", "who", "adult_male", "deck", "embark_town"],
    )

def _load_titanic_medium() -> DatasetConfig:
    """Titanic dataset with heavier nulls - medium cleaning."""
    cfg = _load_titanic_easy()
    rng = np.random.RandomState(123)
    df = cfg.df.copy()

    # Inject extra nulls into numeric columns
    for col in ["age", "fare", "sibsp"]:
        if col in df.columns:
            mask = rng.rand(len(df)) < 0.25
            df.loc[mask, col] = np.nan

    return DatasetConfig(
        df=df,
        target_column="survived",
        exclude_columns=cfg.exclude_columns,
    )

def _load_fe_easy() -> DatasetConfig:
    """Simple dataset for feature-engineering practice (Iris dataset)."""
    df = sns.load_dataset("iris")
    return DatasetConfig(df=df, target_column="species")

def _load_fe_medium() -> DatasetConfig:
    """Medium dataset - more features, categorical data (Penguins dataset)."""
    df = sns.load_dataset("penguins")
    return DatasetConfig(df=df, target_column="species")

def _load_hard_combined() -> DatasetConfig:
    """Hard dataset with nulls AND need for FE - multi-task pipelines (MPG dataset)."""
    df = sns.load_dataset("mpg")
    rng = np.random.RandomState(77)
    
    # Inject nulls into numeric columns
    for col in ["displacement", "horsepower", "weight"]:
        mask = rng.rand(len(df)) < 0.20
        df.loc[mask, col] = np.nan

    # Inject nulls into a categorical column
    mask_cat = rng.rand(len(df)) < 0.10
    df.loc[mask_cat, "origin"] = np.nan

    return DatasetConfig(
        df=df,
        target_column="mpg",
        exclude_columns=["name"]  # Exclude raw string names
    )

# -----------------------------------------------------------------------------
# Registry mapping (TaskType, Difficulty) -> loader callable
# -----------------------------------------------------------------------------

_REGISTRY: Dict[Tuple[TaskType, Difficulty], Callable[[], DatasetConfig]] = {
    # Data Cleaning
    (TaskType.DATA_CLEANING, Difficulty.EASY): _load_titanic_easy,
    (TaskType.DATA_CLEANING, Difficulty.MEDIUM): _load_titanic_medium,
    (TaskType.DATA_CLEANING, Difficulty.HARD): _load_hard_combined,
    # Feature Engineering
    (TaskType.FEATURE_ENGINEERING, Difficulty.EASY): _load_fe_easy,
    (TaskType.FEATURE_ENGINEERING, Difficulty.MEDIUM): _load_fe_medium,
    (TaskType.FEATURE_ENGINEERING, Difficulty.HARD): _load_hard_combined,
}