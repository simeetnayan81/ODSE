"""Abstract base class for all ODSE tasks.

Every concrete task (cleaning, feature engineering, ...) inherits from
``BaseTask`` and implements the abstract hooks. The base class provides:

* ``setup()`` - initialises / resets the state.
* ``execute()`` =. validates + dispatches an action, computes reward.
* ``build_observation()`` - constructs the ``Observation`` pydantic model.
* ``calculate_reward()`` - sklearn proxy model accuracy (5-fold CV).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from ..data.data_manager import DataState
from ..data.datasets import DatasetConfig
from ..models import (
    Action,
    ColumnInfo,
    Difficulty,
    Observation,
    StepResult,
    SubmitAction,
    TaskType,
)

class BaseTask(ABC):

    """Base class that every ODSE task must inherit from.
    
    Subclass contract
    -----------------
    * ``TASK_TYPE``: the ``TaskType`` enum member.
    * ``SUPPORTED_ACTIONS``: set of action_types literal strings.
    * ``apply_action()``: mutate ``self.data_state`` for a given action.
    * ``calculate_reward()``: scaler reward for one step.
    * ``is_done()``: whether the episode should terminate.
    * ``grade()``: score the final state (dict with ``"score"``).
    * ``get_goal_description()``: human readable goal string
    """

    # -- subclasses must set these class-level attributes -------------------
    TASK_TYPE: TaskType
    SUPPORTED_ACTIONS: Set[str]

    #Default per-difficulty step limits (overridable per task)
    MAX_STEPS : Dict[Difficulty, int] = {
        Difficulty.EASY: 15,
        Difficulty.MEDIUM: 30,
        Difficulty.HARD: 45
    }

    def __init__(
        self,
        dataset_config: DatasetConfig,
        difficulty: Difficulty,
        seed: int = 42,
    ) -> None:
        self.dataset_config = dataset_config
        self.difficulty = difficulty
        self.seed = seed
        self.max_steps: int = self.MAX_STEPS.get(difficulty, 25)

        #State - populated on setup()
        self.data_state: DataState | None = None
        self.step_count: int = 0
        self._initial_accuracy: float = 0.0
        self._previous_accuracy: float = 0.0

    # Lifecycle

    def setup(self) -> Observation:
        """Initialize / reset the task. Called by ``ODSEnvironment.reset()``."""
        self.data_state = DataState(self.dataset_config.df, name='initial')
        self.step_count = 0
        self._initial_accuracy = self.calculate_accuracy()
        self._previous_accuracy = self._initial_accuracy
        return self.build_observation()
    
    # Action Dispatch

    def execute(self, action: Action) -> StepResult:
        """Validate, apply, *action*, compute reward, and return ``StepResult``."""
        assert self.data_state is not None, "Call setup() before executing actions"
        self.step_count += 1

        # ----- Submit terminates immediately ------
        if isinstance(action, SubmitAction):
            obs = self.build_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"reason": "Submit"}
            )
        
        # ----- Validate action type for this task -----
        if action.action_type not in self.SUPPORTED_ACTIONS:
            obs = self.build_observation()
            return StepResult(
                observation=obs,
                reward=-0.1,  # Penalty for invalid action
                done=False,
                info={
                    "error": (
                        f"Action `{action.action_type}` not supported "
                        f"by {self.TASK_TYPE.value}"
                    )
                }
            )
        #can be optimized with prev acc
        old_accuracy = self.calculate_accuracy()
        
        # ---- Delegate to the concreate task ----------
        self.apply_action(action)

        new_accuracy = self.calculate_accuracy()

        reward = self.calculate_reward(old_accuracy, new_accuracy, action)
        self._previous_accuracy = new_accuracy
        
        obs = self.build_observation()
        done = self.is_done()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "old_accuracy": old_accuracy,
                "new_accuracy": new_accuracy,
                "accuracy_delta": new_accuracy - old_accuracy
            }
        )

    # Abstract interface - subclasses must implement these methods

    @abstractmethod
    def apply_action(self, action: Action) -> None:
        """Mutate ``self.data_state`` according to *action*"""
        pass

    @abstractmethod
    def calculate_reward(self, old_accuracy: float, new_accuracy: float, action: Action) -> float:
        """Return the scalar reward for a single step"""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """REturn ``True`` when the epiisode should end (besides submit)."""
        pass

    @abstractmethod
    def grade(self) -> Dict[str, Any]:
        """Score the final state. Must return dict containiing ``"score"``."""
        pass

    @abstractmethod
    def get_goal_description(self) -> str:
        """Return a human-readable string describing the agent's goal."""
        pass

    # Observation Builder

    def build_observation(self) -> Observation:
        """Construct an ``Observation`` from the current ``DataState``."""
        
        ds = self.data_state
        df = ds.df

        columns: List[ColumnInfo] = []
        total_nulls = 0
        for col in df.columns:
            nc = int(df[col].isnull().sum())
            total_nulls += nc
            columns.append(
                ColumnInfo(
                    name=col,
                    dtype=str(df[col].dtype),
                    null_count=nc,
                    null_percentage=(
                        round(nc / len(df) * 100, 2) if len(df) > 0 else 0.0
                    ),
                    unique_count=int(df[col].nunique()),
                    is_numeric=pd.api.types.is_numeric_dtype(df[col]),
                    )

            )

        return Observation(
            columns=columns,
            sample_head=df.head().to_dict(orient="list"),
            shape=tuple(df.shape),
            current_accuracy=self.calculate_accuracy(),
            step_count=self.step_count,
            nulls_remaining=total_nulls,
            task_type=self.TASK_TYPE.value,
            difficulty=self.difficulty.value,
            goal_description=self.get_goal_description(),
            available_actions=sorted(self.SUPPORTED_ACTIONS | {"submit"})
        )
    
    # =====================================================================
    # Proxy model accuracy  (shared by all tasks)
    # =====================================================================

    def calculate_accuracy(self) -> float:
        """5-fold CV accuracy using a LogisticRegression proxy model.
        
        Only complete (non-null) rows are used.  Returns 0.0 when the
        dataset is too small or the model fails.
        """
        df = self.data_state.df.dropna()
        target = self.dataset_config.target_column
        features = [
            c for c in self.dataset_config.feature_columns if c in df.columns
        ]

        if len(df) < 10 or target not in df.columns or not features:
            return 0.0

        try:
            X = df[features].copy()
            y = df[target]

            # Encode categorical / string columns
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            X_arr = X.values.astype(float)
            n_folds = min(5, len(df))

            model = LogisticRegression(
                max_iter=1000,
                random_state=self.seed,
                solver="lbfgs",
            )

            scores = cross_val_score(
                model, X_arr, y, cv=n_folds, scoring="accuracy",
            )

            return float(np.mean(scores))
        except Exception:
            return 0.0