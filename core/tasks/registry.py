"""Task registry maps (TaskType, Difficulty) -> concrete task class.

Usage::

    from core.tasks.registry import create_task
    task = create_task("data_cleaning", "easy", seed=42)
    obs  = task.setup()

Third-party extensions can call ``register_task`` to plug in new task types.
"""

from __future__ import annotations

from typing import Dict, Type

from ..data.datasets import DatasetConfig, load_dataset
from ..models import Difficulty, TaskType
from .base_task import BaseTask
from .cleaning_task import CleaningTask
from .feature_engineering_task import FeatureEngineeringTask

# -----------------------------------------------------------------------------
# Internal registry
# -----------------------------------------------------------------------------

_TASK_CLASSES: Dict[TaskType, Type[BaseTask]] = {
    TaskType.DATA_CLEANING: CleaningTask,
    TaskType.FEATURE_ENGINEERING: FeatureEngineeringTask,
}

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def create_task(
    task_type: TaskType | str,
    difficulty: Difficulty | str = Difficulty.EASY,
    *,
    seed: int = 42,
    dataset_config: DatasetConfig | None = None,
) -> BaseTask:
    """Factory: create a task instance ready for ``setup()``.

    Parameters
    ----------
    task_type
        Which task to create (enum or string value).
    difficulty
        Difficulty level (enum or string value).
    seed
        RNG seed for reproducibility.
    dataset_config
        Optional override; if *None* the default dataset for the
        ``(task_type, difficulty)`` pair is loaded from the registry.

    Returns
    -------
    BaseTask
        A fully-configured (but not yet initialised) task instance.
        Call ``.setup()`` to get the first observation.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    if isinstance(difficulty, str):
        difficulty = Difficulty(difficulty)

    cls = _TASK_CLASSES.get(task_type)
    if cls is None:
        raise ValueError(f"Unknown task type: {task_type!r}")

    if dataset_config is None:
        dataset_config = load_dataset(task_type, difficulty)

    return cls(dataset_config=dataset_config, difficulty=difficulty, seed=seed)

def register_task(task_type: TaskType, cls: Type[BaseTask]) -> None:
    """Register a new task class (for third-party extensions).

    Example::

        from core.models import TaskType
        from core.tasks.registry import register_task

        class MyCustomTask(BaseTask):
            ...

        register_task(TaskType("my_custom"), MyCustomTask)
    """
    _TASK_CLASSES[task_type] = cls