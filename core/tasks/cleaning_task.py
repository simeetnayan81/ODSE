"""Data Cleaning task implementation

Supported actions: ``impute``, ``drop_columns``, ``drop_rows``, ``submit``.

Reward logic:
    reward = (new_accuracy - old_accuracy)*10 - 0.01 (step_penalty)

Termination:
    * All null values are eliminated
    * Maximum step count reached
    TODO: Work on further termination conditions for different types of data
"""

from __future__ import annotations

from typing import Any, Dict, Set

import pandas as pd

from ..models import (
    Action,
    Difficulty,
    DropColumnAction,
    DropRowAction,
    ImputeAction,
    TaskType,
    )

from .base_task import BaseTask


class CleaningTask(BaseTask):
    """Clean a dirty dataset by removing / imputing missing values."""

    TASK_TYPE = TaskType.DATA_CLEANING
    SUPPORTED_ACTIONS: Set[str] = {"impute", "drop_columns", "drop_rows"}

    # Action application

    def apply_action(self, action: Action) -> None: #noqa: D401
        df= self.data_state.df.copy()

        if isinstance(action, ImputeAction):
            df = self._impute(
                df, action.column, action.strategy, action.fill_value
            )
            label = f"impute({action.column}, {action.strategy})"
        
        elif isinstance(action, DropColumnAction):
            if action.column in df.columns:
                df = df.drop(columns=[action.column])
            label = f"drop_column({action.column})"

        elif isinstance(action, DropRowAction):
            if action.column in df.columns:
                df = df.dropna(subset=[action.column])
            label = f"drop_row({action.column})"
        
        else:
             return # unsupported action, should not happen due to prior validation

        self.data_state.apply_update(df, action_name=label)

    #Reward

    def calculate_reward(self, old_accuracy: float, new_accuracy: float, action: Action) -> float:
        accuracy_gain=new_accuracy-old_accuracy
        return accuracy_gain*10 - 0.01
    
    #Termination

    def is_done(self) -> bool:
        #TODO: Add more termination conditions
        #Think more about total_nulls, maybe remove this condition, 
        #If dataset has no rows/columns left, end the episode
        if self.data_state.total_nulls == 0:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False
    
    # Grading


    def grade(self) -> Dict[str, Any]:
        score = 0.0
        details: Dict[str, Any] = {}
        
        # 50% - no nulls remaining
        nulls = self.data_state.total_nulls
        if nulls == 0:
            score += 0.5
            details["nulls_check"] = "passed"
        else:
            details["nulls_remaining"] = "failed"

        
        # 50% - accuracy above threshold (scales with difficulty)
        accuracy = self.calculate_accuracy()
        details["final_accuracy"] = round(accuracy, 4)
        threshold = {
            Difficulty.EASY: 0.75,
            Difficulty.MEDIUM: 0.80,
            Difficulty.HARD: 0.85
        }.get(self.difficulty, 0.80)
        if accuracy > threshold:
            score += 0.5
            details["accuracy_check"] = "passed"
        else:
            details["accuracy_check"] = "failed"
        
        details["score"] = min(score, 1.0)
        details["steps_taken"] = self.step_count
        details["action_history"] = list(self.data_state.history)
        return details

    # Goal


    def get_goal_description(self) -> str:
        return (
            "DATA CLEANINING: Remove all missing values from the dataset while "
            "maintaining or improving model accuracy. Use impute, drop_column, "
            "or drop_rows actions. Submit when finished."
        )
    
    # Helpers

    @staticmethod
    def _impute(
        df: pd.DataFrame, 
        column: str, 
        strategy: str, 
        fill_value: Any = None
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df
        df = df.copy()
        imputed_value = None
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
            imputed_value = df[column].mean()
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
            imputed_value = df[column].median()
        elif strategy == "mode":
            imputed_value = df[column].mode()
            if len(imputed_value) > 0:
                imputed_value = imputed_value.iloc[0]
        elif strategy == "constant":
            if fill_value is not None:
                imputed_value = fill_value
        else:
            # Unsupported strategy or non-numeric column for mean/median
            return df
        if imputed_value is not None:
            df[column] = df[column].fillna(imputed_value)
        
        return df