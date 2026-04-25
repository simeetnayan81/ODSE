"""
OpenEnv-compatible data models for the ODSE (Open Data Science Environment).

Three types are defined:

* OdseAction      - action with action_type and optional code.
* OdseObservation - extends OpenEnv Observation with all ODSE fields.
                    The core Observation is kept free of OpenEnv dependencies;
                    this class bridges the two.
* OdseState       - extends OpenEnv State with data-science episode metadata.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation as OpenEnvObservation, State
from pydantic import Field

from odse.core.models import DatasetInfo, ExecutionStatus, VariableInfo

# ==============================================================================
# Action
# ==============================================================================

class OdseAction(Action):
    """Action for the ODSE environment.

    Two action types are supported:

    * run_code - execute Python code in the persistent sandbox.
    * submit   - submit predictions and end the episode.
    """

    action_type: Literal["run_code", "submit"] = Field(
        ..., description="Type of action: 'run_code' or 'submit'"
    )
    code: Optional[str] = Field(
        default=None,
        description="Python code to execute (required when action_type='run_code')",
    )

# ==============================================================================
# Observation
# ==============================================================================

class OdseObservation(OpenEnvObservation):
    """OpenEnv-compatible observation for the ODSE environment.

    Extends OpenEnv Observation (done, reward, metadata) with all
    ODSE-specific fields. The core Observation has no OpenEnv dependency;
    the server wrapper constructs this class from core output.
    """

    # -- Execution result (empty on reset) -------------------------------------
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

    # -- Workspace state -------------------------------------------------------
    namespace_summary: List[VariableInfo] = Field(
        default_factory=list,
        description="User-visible variables in the sandbox namespace",
    )

    # -- Scoring ---------------------------------------------------------------
    validation_score: Optional[float] = Field(
        default=None,
        description="Latest validation score (from evaluate or auto-detected)",
    )
    best_validation_score: Optional[float] = Field(
        default=None,
        description="Best validation score achieved this episode",
    )
    test_score: Optional[float] = Field(
        default=None,
        description="Test-set score (populated only after submit)",
    )
    test_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full test evaluation report (populated only after submit)",
    )

    # -- Episode context -------------------------------------------------------
    step_count: int = Field(
        default=0, ge=0, description="Steps taken so far",
    )
    max_steps: int = Field(
        default=20, ge=1, description="Step budget for this episode",
    )
    dataset_info: Optional[DatasetInfo] = Field(
        default=None, description="Metadata about the dataset",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the agent objective",
    )

    # -- Step diagnostics (named field - survives serialize_observation) -------
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step diagnostics (scores, timing, breakdown)",
    )

# ==============================================================================
# State
# ==============================================================================

class OdseState(State):
    """Episode state for the ODSE environment."""

    dataset_name: str = Field(
        default="", description="Name of the dataset (e.g. 'breast_cancer', 'iris')"
    )
    difficulty: str = Field(
        default="easy", description="Difficulty level: 'easy', 'medium', or 'hard'"
    )
    problem_type: str = Field(
        default="", description="'classification' or 'regression'"
    )
    target_column: str = Field(
        default="", description="Name of the target column in the dataset"
    )
    problem_description: str = Field(
        default="", description="Human-readable description of the dataset objective"
    )
    metric: str = Field(
        default="", description="Primary evaluation metric (e.g. 'accuracy', 'r2')"
    )
    max_steps: int = Field(
        default=20, ge=1, description="Maximum code-execution steps for this episode"
    )
    done: bool = Field(
        default=False, description="Whether the episode has ended"
    )
    best_validation_score: Optional[float] = Field(
        default=None,
        description="Best validation score achieved so far this episode",
    )
    latest_validation_score: Optional[float] = Field(
        default=None,
        description="Most recent validation score",
    )