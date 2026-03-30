import pandas as pd
from typing import Dict, Any, List

'''
DataState class to manage the current state of the dataframe, its metadata, and history of actions taken.
This class provides a structured way to access the dataframe and its metadata, and to apply updates while 
keeping track of the history of transformations. Currently, it reads data from a CSV file, but it can be extended
to support other data sources in the future(links, databases, etc.). 
#TODO: Add support for loading from different data sources, and for saving the state to disk.
'''
class DataState:
    def __init__(self, df: pd.DataFrame, name: str = "initial"):
        self._df = df.copy()
        self.name = name
        self.history: List[str] = []
        
        # Cache metadata to avoid recalculating every time
        self.metadata = self._generate_metadata()

    def _generate_metadata(self) -> Dict[str, Any]:
        """Calculates internal stats for the Observation."""
        return {
            "shape": self._df.shape,
            "null_counts": self._df.isnull().sum().to_dict(),
            "dtypes": self._df.dtypes.apply(lambda x: str(x)).to_dict(),
            "columns": list(self._df.columns)
        }

    @property
    def df(self) -> pd.DataFrame:
        """Read-only access to the dataframe."""
        return self._df

    def apply_update(self, new_df: pd.DataFrame, action_name: str):
        """Updates the state and logs the history."""
        self._df = new_df
        self.history.append(action_name)
        self.metadata = self._generate_metadata()

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Returns specific stats for a column."""
        if column not in self._df.columns:
            return {}
        return {
            "unique_count": self._df[column].nunique(),
            "is_numeric": pd.api.types.is_numeric_dtype(self._df[column])
        }