"""DataState: wrapper around a pandas DataFrame with cached metadata and history.

The underlying DataFrame is the single source of truth. Metadata is
re-computed after every mutation via ``apply_update``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


class DataState:
    """Manages the current state of a dataset with cached metadata and action history.

    Parameters
    ----------
    df : pd.DataFrame
        The initial dataset (a copy is stored internally).
    name : str
        A human-readable label for this state snapshot.
    """

    def __init__(self, df: pd.DataFrame, *, name: str = "initial") -> None:
        self._df: pd.DataFrame = df.copy()
        self.name: str = name
        self.history: List[str] = []
        self.metadata: Dict[str, Any] = self._generate_metadata()

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def _generate_metadata(self) -> Dict[str, Any]:
        """Calculate summary statistics for the current dataframe."""
        return {
            "shape": self._df.shape,
            "null_counts": self._df.isnull().sum().to_dict(),
            "total_nulls": int(self._df.isnull().sum().sum()),
            "dtypes": self._df.dtypes.apply(str).to_dict(),
            "columns": list(self._df.columns),
            "numeric_columns": list(
                self._df.select_dtypes(include="number").columns,
            ),
            "categorical_columns": list(
                self._df.select_dtypes(include=["object", "category"]).columns,
            ),
        }

    # -------------------------------------------------------------------------
    # Read-only properties
    # -------------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Read-only access to the underlying dataframe."""
        return self._df

    @property
    def total_nulls(self) -> int:
        """Total number of null values across the entire dataset."""
        return self.metadata["total_nulls"]

    @property
    def shape(self) -> tuple:
        """``(rows, columns)`` of the dataset."""
        return self.metadata["shape"]

    @property
    def columns(self) -> List[str]:
        """Column names in the dataset."""
        return self.metadata["columns"]

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def apply_update(self, new_df: pd.DataFrame, action_name: str) -> None:
        """Replace the internal dataframe, log the action, and refresh metadata."""
        self._df = new_df
        self.history.append(action_name)
        self.metadata = self._generate_metadata()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Return quick stats for a single column."""
        if column not in self._df.columns:
            return {}

        col = self._df[column]
        return {
            "null_count": int(col.isnull().sum()),
            "unique_count": int(col.nunique()),
            "is_numeric": pd.api.types.is_numeric_dtype(col),
            "dtype": str(col.dtype),
        }

    def clone(self, name: Optional[str] = None) -> "DataState":
        """Return a deep copy of this state."""
        new_state = DataState(self._df, name=name or self.name)
        new_state.history = list(self.history)
        return new_state