"""Feature-Engineering task implementation.

Supported actions: ``create_interaction``, ``bin_column``,
``one_hot_encode``, ``scale_column``, ``log_transform``, ``submit``.

Reward logic:
    reward = (new_accuracy - old_accuracy) * 15 - 0.01 (step penalty)

Termination:
    * The agent calls ``SubmitAction``, **or**
    * The maximum step count is reached.
"""

from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from ..models import (
    Action,
    BinColumnAction,
    CreateInteractionAction,
    Difficulty,
    LogTransformAction,
    OneHotEncodeAction,
    ScaleColumnAction,
    TaskType,
)
from .base_task import BaseTask


class FeatureEngineeringTask(BaseTask):
    """Engineer new features to improve model accuracy."""

    TASK_TYPE = TaskType.FEATURE_ENGINEERING
    SUPPORTED_ACTIONS: Set[str] = {
        "create_interaction",
        "bin_column",
        "one_hot_encode",
        "scale_column",
        "log_transform",
    }

    # -------------------------------------------------------------------------
    # Action application
    # -------------------------------------------------------------------------

    def apply_action(self, action: Action) -> None:  # noqa: D401
        df = self.data_state.df.copy()

        if isinstance(action, CreateInteractionAction):
            df = self._create_interaction(df, action)
            label = f"interaction({action.column_a}*{action.column_b})"

        elif isinstance(action, BinColumnAction):
            df = self._bin_column(df, action)
            label = f"bin({action.column}, n={action.n_bins})"

        elif isinstance(action, OneHotEncodeAction):
            df = self._one_hot_encode(df, action)
            label = f"ohe({action.column})"

        elif isinstance(action, ScaleColumnAction):
            df = self._scale_column(df, action)
            label = f"scale({action.column}, {action.method})"

        elif isinstance(action, LogTransformAction):
            df = self._log_transform(df, action)
            label = f"log1p({action.column})"

        else:
            return

        # --- keep the feature list in sync -------------------------
        new_cols = set(df.columns) - set(self.data_state.df.columns)
        for c in new_cols:
            if (
                c != self.dataset_config.target_column
                and c not in self.dataset_config.exclude_columns
                and c not in self.dataset_config.feature_columns
            ):
                self.dataset_config.feature_columns.append(c)

        # Remove columns that were dropped (e.g. OHE drop_original)
        self.dataset_config.feature_columns = [
            c for c in self.dataset_config.feature_columns if c in df.columns
        ]

        self.data_state.apply_update(df, label)

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------

    def calculate_reward(
        self,
        old_accuracy: float,
        new_accuracy: float,
        action: Action,
    ) -> float:
        accuracy_gain = new_accuracy - old_accuracy
        return accuracy_gain * 15.0 - 0.01

    # -------------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------------

    def is_done(self) -> bool:
        # FE tasks only auto-terminate on max steps (otherwise via submit)
        return self.step_count >= self.max_steps

    # -------------------------------------------------------------------------
    # Grading
    # -------------------------------------------------------------------------

    def grade(self) -> Dict[str, Any]:
        accuracy = self.calculate_accuracy()
        improvement = accuracy - self._initial_accuracy

        details: Dict[str, Any] = {
            "initial_accuracy": round(self._initial_accuracy, 4),
            "final_accuracy": round(accuracy, 4),
            "improvement": round(improvement, 4),
            "features_created": len(self.data_state.history),
            "steps_taken": self.step_count,
            "action_history": list(self.data_state.history),
        }

        # Score bands based on relative improvement
        if improvement >= 0.10:
            score = 1.0
        elif improvement >= 0.05:
            score = 0.75
        elif improvement >= 0.02:
            score = 0.5
        elif improvement > 0:
            score = 0.25
        else:
            score = 0.0

        details["score"] = score
        return details

    # -------------------------------------------------------------------------
    # Goal
    # -------------------------------------------------------------------------

    def get_goal_description(self) -> str:
        return (
            "FEATURE ENGINEERING: Create new features to improve model accuracy. "
            "Use create_interaction, bin_column, one_hot_encode, scale_column, "
            "or log_transform actions. Submit when finished."
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_interaction(
        df: pd.DataFrame,
        action: CreateInteractionAction,
    ) -> pd.DataFrame:
        if action.column_a not in df.columns or action.column_b not in df.columns:
            return df
        a, b = df[action.column_a], df[action.column_b]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            df = df.copy()
            df[action.new_column] = a * b
        return df

    @staticmethod
    def _bin_column(df: pd.DataFrame, action: BinColumnAction) -> pd.DataFrame:
        if action.column not in df.columns:
            return df
        col = df[action.column]
        if not pd.api.types.is_numeric_dtype(col):
            return df
        df = df.copy()
        new_col = f"{action.column}_binned"
        try:
            if action.strategy == "quantile":
                df[new_col] = pd.qcut(
                    col, q=action.n_bins, labels=False, duplicates="drop",
                )
            else:
                df[new_col] = pd.cut(col, bins=action.n_bins, labels=False)
        except Exception:
            pass  # graceful no-op on degenerate data
        return df

    @staticmethod
    def _one_hot_encode(
        df: pd.DataFrame,
        action: OneHotEncodeAction,
    ) -> pd.DataFrame:
        if action.column not in df.columns:
            return df
        dummies = pd.get_dummies(df[action.column], prefix=action.column)
        df = pd.concat([df, dummies], axis=1)
        if action.drop_original:
            df = df.drop(columns=[action.column])
        return df

    @staticmethod
    def _scale_column(
        df: pd.DataFrame,
        action: ScaleColumnAction,
    ) -> pd.DataFrame:
        if action.column not in df.columns:
            return df
        col = df[action.column]
        if not pd.api.types.is_numeric_dtype(col):
            return df
        df = df.copy()
        if action.method == "standard":
            std = col.std()
            if std > 0:
                df[action.column] = (col - col.mean()) / std
        elif action.method == "minmax":
            cmin, cmax = col.min(), col.max()
            if cmax > cmin:
                df[action.column] = (col - cmin) / (cmax - cmin)
        return df

    @staticmethod
    def _log_transform(
        df: pd.DataFrame,
        action: LogTransformAction,
    ) -> pd.DataFrame:
        if action.column not in df.columns:
            return df
        col = df[action.column]
        if not pd.api.types.is_numeric_dtype(col):
            return df
        df = df.copy()
        df[action.column] = np.log1p(np.abs(col))
        return df