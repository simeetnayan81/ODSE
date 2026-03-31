"""Data management with train / validation / test splits.

Provides the ``DataSplit`` dataclass that partitions a dataset into
three non-overlapping subsets with proper isolation:

* **train** : full features + target (visible to the agent)
* **val** : features only (target hidden, used by ``evaluate()``)
* **test** : features only (target hidden, used on ``Submit``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from .datasets import DatasetConfig

@dataclass
class DataSplit:
    """A partitioned dataset with proper isolation.

    Agent-visible
    -------------
    * ``train_df``      : training data with features **and** target
    * ``val_features``  : validation features (no target)
    * ``test_features`` : test features (no target)

    Hidden (used internally by the environment)
    -------------------------------------------
    * ``val_labels``    : validation targets (for ``evaluate()``)
    * ``test_labels``   : test targets (for final scoring on ``Submit``)
    """

    train_df: pd.DataFrame
    val_features: pd.DataFrame
    val_labels: pd.Series
    test_features: pd.DataFrame
    test_labels: pd.Series


def create_data_split(
    config: "DatasetConfig",
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Split a :class:`DatasetConfig` into train / val / test.

    Parameters
    ----------
    config
        Dataset with ``df``, ``target_column``, ``feature_columns``.
    val_ratio
        Fraction of data reserved for validation.
    test_ratio
        Fraction of data reserved for the hidden test set.
    seed
        Random seed for reproducible splits.

    Returns
    -------
    DataSplit
    """
    df = config.df.copy()
    target = config.target_column

    # 1) Separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # 2) Separate validation from remaining train
    relative_val = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        random_state=seed,
        shuffle=True,
    )

    # Reset indices for cleanliness
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Determine feature columns present in the data
    feature_cols: List[str] = [
        c for c in config.feature_columns if c in df.columns
    ]

    return DataSplit(
        train_df=train_df,                       # full (features + target)
        val_features=val_df[feature_cols],       # features only
        val_labels=val_df[target],               # hidden
        test_features=test_df[feature_cols],     # features only
        test_labels=test_df[target],             # hidden
    )