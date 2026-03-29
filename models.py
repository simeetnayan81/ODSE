"""Pydantic models for ODSE (Open Data Science Environment)."""

from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ImputeAction(BaseModel):
    """Impute missing values in a column using a specified strategy."""
    action_type: Literal["impute"] = Field(default="impute", frozen=True)
    column: str = Field(description="Name of the column to impute")
    strategy: Literal["mean", "median", "mode"] = Field(
        description="Strategy for imputation"
    )


class DropAction(BaseModel):
    """Drop a column from the dataset."""
    action_type: Literal["drop"] = Field(default="drop", frozen=True)
    column: str = Field(description="Name of the column to drop")


class SubmitAction(BaseModel):
    """Submit the current state and terminate the episode."""
    action_type: Literal["submit"] = Field(default="submit", frozen=True)


# Discriminated union of all possible actions
Action = Annotated[
    ImputeAction | DropAction | SubmitAction,
    Field(discriminator="action_type")
]


class Observation(BaseModel):
    """Observation returned by the environment.
    
    Includes metadata about the current dataset state and performance metrics.
    """
    column_metadata: Dict[str, Dict[str, Any]] = Field(
        description="Metadata for each column: null_count, type, and null_percentage"
    )
    sample_head: Dict[str, list] = Field(
        description="JSON representation of first 5 rows of the dataset"
    )
    current_accuracy: float = Field(
        description="Current model accuracy via 5-fold CV on LogisticRegression",
        ge=0.0,
        le=1.0
    )
    step_count: int = Field(
        description="Number of steps taken in the episode",
        ge=0
    )
    nulls_remaining: int = Field(
        description="Total number of null values remaining in the dataset",
        ge=0
    )


class StepResult(BaseModel):
    """Result returned by the step() method."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
