"""Pydantic models for ODSE (Open Data Science Environment).

Defines all action types, observations, step results, and configuration enums,
Actions use a discriminated union pattern for type safety dispatch.

To add a new action type:
    1. Define a new ```BaseModel``` with a unique `action_type` Literal.
    2. Add it to the `Action` union at the bottom of this file.
    3. Register it in the relevant task's ``SUPPORTED_ACTIONS`` set.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple
from pydantic import BaseModel, Field
from typing_extensions import Annotated

class Difficulty(str, Enum):
    """Difficulty levels for tasks and datasets."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class TaskType(str, Enum):
    """Available task types in the ODSE.

    Extend this enum when adding new task categories.
    """

    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    #Future Extensions
    #MODEL_TRAINING = "model_training"


# Actions(Data Cleaning)
 
class ImputeAction(BaseModel):
    """Impute missing values in a column using a specified strategy."""

    action_type: Literal["impute"] = Field(default="impute", frozen=True)
    column: str = Field(description="Name of the column to impute")
    strategy: Literal["mean", "median", "mode", "constant"] = Field(
        description="Imputation strategy to use"
    )
    fill_value: Optional[Any] = Field(
        default=None,
        description="Value to use when strategy is 'constant'"
    )

class DropColumnAction(BaseModel):
    """Drop an entire column from the dataset."""

    action_type: Literal["drop_column"] = Field(default="drop_column", frozen=True)
    column: str = Field(description="Name of the column to drop")

class DropRowAction(BaseModel):
    """Drop rows that contain null values in a specific column."""
    
    action_type: Literal["drop_row"] = Field(default="drop_row", frozen=True)
    column: str = Field(description="Column whose null rows should be dropped")

#
# Actions(Feature Engineering)
#

class CreateInteractionAction(BaseModel):
    """Create a new feature by multiplying two existing features."""

    action_type: Literal["create_interaction"] = Field(
        default="create_interaction", 
        frozen=True
    )
    column_a: str = Field(description="First column")
    column_b: str = Field(description="Second column")
    new_column: str = Field(description="Name of the new interaction feature")

class BinColumnAction(BaseModel):
    """Discretize a numeric column into categorical bins."""

    action_type: Literal["bin_column"] = Field(default="bin_column", frozen=True)
    column: str = Field(description="Name of the column to bin")
    n_bins: int = Field(description="Number of bins to create")
    strategy: Literal['uniform', 'quantile', 'kmeans'] = Field(
        default='quantile',
        description="Binning strategy"
    )

class OneHotEncodeAction(BaseModel):
    """One-hot encode a categorical column."""

    action_type: Literal["one_hot_encode"] = Field(
        default="one_hot_encode", 
        frozen=True
    )
    column: str = Field(description="Categorical column to encode")
    drop_original: bool = Field(
        default=False,
        description="Whether to drop the original column after encoding"
    )

class ScaleColumnAction(BaseModel):
    """Scale a numeric column (standardize or min-max normalise)."""

    action_type: Literal["scale_column"] = Field(default="scale_column", frozen=True)
    column: str = Field(description="Numeric column to scale")
    method: Literal["standard", "minmax"] = Field(
        default="standard",
        description="Scaling method"
    )

class LogTransformAction(BaseModel):
    """Apply log(1 + |x|) transformation to a numeric column."""

    action_type: Literal["log_transform"] = Field(
        default="log_transform", 
        frozen=True
    )
    column: str = Field(description="Numeric column to log-transform")


#  Common Actions

class SubmitAction(BaseModel):
    """Submit the current state and terminate the episode."""

    action_type: Literal["submit"] = Field(default="submit", frozen=True)



# Action Union (discriminated)
# To register a new action: define the model above and add it here

Action = Annotated[
    Union[
        #Cleaning
        ImputeAction,
        DropColumnAction,
        DropRowAction,
        #Feature Engineering
        CreateInteractionAction,
        BinColumnAction,
        OneHotEncodeAction,
        ScaleColumnAction,
        LogTransformAction,
        #Common
        SubmitAction
    ],
    Field(discriminator="action_type")
]


# Column Metadata (embedded inside Observation)

class ColumnInfo(BaseModel):
    """Metadata for a single column in the dataset."""

    name: str
    dtype: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0, le=100)
    is_numeric: bool
    unique_count: int = Field(ge=0)
    #Future Extensions
    #mean: Optional[float] = None
    #std: Optional[float] = None
    #min: Optional[float] = None
    #max: Optional[float] = None 

class Observation(BaseModel):
    """Observation returned by the environment.
    Provides the agent with everything it needs to decide its next action.
    """

    columns: List[ColumnInfo] = Field(
        description = "Metadata for every column in the current dataset state"
    )
    sample_head: Dict[str, list] = Field(
        description="First 5 rows of the dataset as {col: [values]}"
    )
    shape: Tuple[int, int] = Field(
        description="Shape of the dataset (rows, columns)"
    )
    current_accuracy: float = Field(
        ge=0.0, le=1.0, 
        description="Proxy model accuracy (5-fold CV)"
    )
    step_count: int = Field(description = "Steps taken so far", ge=0)
    nulls_remaining: int = Field(ge=0,  description="Total number of null values remaining in the dataset")
    task_type: str = Field(description="Current task type identifier")
    difficulty: str = Field(description="Current difficulty level")
    goal_description: str = Field(description="Human-readable description of the task goal")
    available_actions: List[str] = Field(
        description="List of action types that are valid in the current state"
    )

# Step result

class StepResult(BaseModel):
    """Result returned by ``env.step()``."""

    observation: Observation
    reward: float = Field(description="Reward received for the action taken")
    done: bool = Field(description="Whether the episode has terminated")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info for debugging or analysis")