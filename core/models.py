"""Pydantic models for the ODSE Sandbox Environment.

Defines the two actions (RunCode, Submit), observations, step results,
and supporting types for a code-execution sandbox where agents write
and execute Python code to solve data-science tasks.

Architecture
------------
Instead of a fixed DSL with enumerated action types, the sandbox
exposes only two actions:

* ``RunCodeAction`` : execute arbitrary Python in a persistent namespace.
* ``SubmitAction``  : submit predictions and terminate the episode.

The observation gives the agent execution feedback (stdout/stderr),
workspace state (variables, shapes), scoring context, and dataset
metadata so it can plan its next code cell.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


# ============================================================================
# Enums
# ============================================================================

class ProblemType(str, Enum):
    """Type of ML problem the agent must solve."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Difficulty(str, Enum):
    """Difficulty Level - controls dataset noise, nulls, and step budget."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExecutionStatus(str, Enum):
    """Outcome of a single code execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ============================================================================
# Actions
# ============================================================================

class RunCodeAction(BaseModel):
    """Execute Python code in the sandbox.

    The code runs in a persistent namespace pre-loaded with:

    * ``train_df``      : Training DataFrame (features **+** target)
    * ``val_features``  : Validation features (target hidden)
    * ``test_features`` : Test features (target hidden)
    * ``target_column`` : Name of the target column (str)
    * ``pd``, ``np``    : pandas and numpy
    * ``evaluate(preds)`` : Score predictions against hidden **validation** labels

    Variables persist across ``RunCode`` calls (notebook-style kernel).
    Assign your **test-set** predictions to the variable ``predictions``
    before calling ``SubmitAction``.
    """

    action_type: Literal["run_code"] = Field(default="run_code", frozen=True)
    code: str = Field(description="Python code to execute in the sandbox")


class SubmitAction(BaseModel):
    """Submit predictions and terminate the episode.

    Reads the ``predictions`` variable from the sandbox namespace and
    scores it against the **hidden test labels**. The variable must be
    an array-like whose length matches ``test_features``.
    """

    action_type: Literal["submit"] = Field(default="submit", frozen=True)


Action = Annotated[
    Union[RunCodeAction, SubmitAction],
    Field(discriminator="action_type"),
]


# ============================================================================
# Dataset / Column metadata
# ============================================================================

class ColumnSchema(BaseModel):
    """Schema information for a single column in the dataset."""

    name: str
    dtype: str
    null_count: int = Field(ge=0)
    is_numeric: bool
    unique_count: int = Field(ge=0)
    sample_values: List[Any] = Field(default_factory=list, max_length=5)


class DatasetInfo(BaseModel):
    """Metadata about the dataset, provided to the agent on every observation."""

    train_shape: Tuple[int, int]
    val_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    target_column: str
    problem_type: str  # "classification" or "regression"
    metric: str  # primary metric name (e.g. "accuracy", "r2")
    columns: List[ColumnSchema]
    target_classes: Optional[List[Any]] = None  # classification only
    target_stats: Optional[Dict[str, float]] = None  # regression only


# ============================================================================
# Namespace summary
# ============================================================================

class VariableInfo(BaseModel):
    """Summary of one variable in the agent's sandbox namespace."""

    name: str
    type_name: str
    shape: Optional[Tuple[int, ...]] = None
    preview: str = Field(default="", max_length=500)


# ============================================================================
# Observation
# ============================================================================

class Observation(BaseModel):
    """Observation returned after every ``reset()`` or ``step()``."""

    # - Execution result (empty on reset) ------------------------------------
    stdout: str = Field(
        default="",
        description="Captured stdout from last code execution",
    )
    stderr: str = Field(
        default="",
        description="Captured stderr / traceback from last execution",
    )
    execution_status: Optional[ExecutionStatus] = Field(
        default=None,
        description="Status of the last code execution (None on reset)",
    )
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time of last execution in milliseconds",
    )

    # - Workspace state ------------------------------------------------------
    namespace_summary: List[VariableInfo] = Field(
        default_factory=list,
        description="User-visible variables in the sandbox namespace",
    )

    # - Scoring --------------------------------------------------------------
    validation_score: Optional[float] = Field(
        default=None,
        description="Latest validation score (from evaluate() or auto-detected)",
    )
    best_validation_score: Optional[float] = Field(
        default=None,
        description="Best validation score achieved this episode",
    )

    # - Episode context ------------------------------------------------------
    step_count: int = Field(ge=0, description="Steps taken so far")
    max_steps: int = Field(ge=1, description="Step budget for this episode")
    dataset_info: DatasetInfo
    task_description: str = Field(
        description="Human-readable description of the agent's objective",
    )
    done: bool = Field(default=False, description="Whether the episode has ended")


# ============================================================================
# Step result
# ============================================================================

class StepResult(BaseModel):
    """Result of a single ``env.step()`` call."""

    observation: Observation
    reward: float = Field(description="Scalar reward for this step")
    done: bool = Field(description="Whether the episode has terminated")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostics (scores, timing, breakdown, ...)",
    )