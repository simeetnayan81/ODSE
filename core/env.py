"""ODSE Environment - the main entry-point for RL agents.

Follows the strict API: ``reset()``, ``state()``, ``step(action)``.
All domain-specific logic is delegated to the active *task* object,
making the environment fully task-agnostic and extensible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .models import (
    Action,
    Difficulty,
    Observation,
    StepResult,
    TaskType,
)
from .tasks.base_task import BaseTask
from .tasks.registry import create_task


class ODSEnvironment:
    """Open Data Science Environment.

    Provides a standardised RL API around data-science tasks
    (cleaning, feature engineering, and future extensions).

    Parameters
    ----------
    task_type
        Which task to run (e.g. ``"data_cleaning"`` or ``TaskType.DATA_CLEANING``).
    difficulty
        ``"easy"``, ``"medium"``, or ``"hard"``.
    seed
        RNG seed for reproducibility.

    Example
    -------
    >>> env = ODSEnvironment(task_type="data_cleaning", difficulty="easy")
    >>> obs = env.reset()
    >>> result = env.step(ImputeAction(column="age", strategy="mean"))
    >>> print(result.reward, result.observation.nulls_remaining)
    """

    def __init__(
        self,
        task_type: TaskType | str = TaskType.DATA_CLEANING,
        difficulty: Difficulty | str = Difficulty.EASY,
        seed: int = 42,
    ) -> None:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if isinstance(difficulty, str):
            difficulty = Difficulty(difficulty)

        self.task_type = task_type
        self.difficulty = difficulty
        self.seed = seed

        self._task: Optional[BaseTask] = None

    # -------------------------------------------------------------------------
    # Public API (reset / state / step)
    # -------------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        np.random.seed(self.seed)
        self._task = create_task(
            self.task_type, self.difficulty, seed=self.seed,
        )
        return self._task.setup()

    def state(self) -> Observation:
        """Return the current observation *without* advancing the episode."""
        self._ensure_task()
        return self._task.build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute *action* and return the step result."""
        self._ensure_task()
        return self._task.execute(action)

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    @property
    def task(self) -> BaseTask:
        """Direct access to the underlying task (useful for grading / inspection)."""
        self._ensure_task()
        return self._task

    @property
    def working_df(self) -> pd.DataFrame:
        """Shortcut to the current working DataFrame."""
        self._ensure_task()
        return self._task.data_state.df

    def grade(self) -> Dict[str, Any]:
        """Grade the current episode and return a detailed report.

        The report always contains ``"score"`` (0.0 - 1.0).
        """
        self._ensure_task()
        return self._task.grade()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _ensure_task(self) -> None:
        if self._task is None:
            raise RuntimeError(
                "Environment not initialised - call reset() first."
            )


# -----------------------------------------------------------------------------
# Standalone grading helper
# -----------------------------------------------------------------------------

def grade_performance(env: ODSEnvironment) -> Dict[str, Any]:
    """Convenience function: grade the current episode.

    Returns a dict with at least ``{"score": float}`` (0.0 - 1.0 range).
    """
    return env.grade()