"""ODSE Sandbox Environment - the main entry-point for RL agents.

Instead of a fixed DSL, agents write and execute Python code in a
persistent, sandboxed namespace. The environment provides:

* Pre-loaded data (``train_df``, ``val_features``, ``test_features``)
* A sandboxed executor with whitelisted imports and time limits
* An ``evaluate(predictions)`` function for validation feedback
* Dense heuristic rewards per step + a final test-set score on submit

API
---
``reset() -> Observation``
    Initialize the episode and return the first observation.

``step(action) -> StepResult``
    Execute a ``RunCodeAction`` or ``SubmitAction``.

``state() -> Observation``
    Read current observation without advancing the episode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .data.data_manager import DataSplit, create_data_split
from .data.datasets import DatasetConfig, load_dataset
from .evaluator import compute_full_report, compute_metric
from .executor import SandboxExecutor
from .models import (
    Action,
    ColumnSchema,
    DatasetInfo,
    Difficulty,
    ExecutionStatus,
    Observation,
    ProblemType,
    RunCodeAction,
    StepResult,
    SubmitAction,
    VariableInfo,
)
from .reward import compute_step_reward, compute_submit_reward


class ODSEnvironment:
    """Open Data Science Sandbox Environment.

    Agents interact by writing and executing Python code. The
    environment provides train/validation data, a sandboxed executor,
    and evaluates predictions on hidden holdout data.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. ``"titanic"``, ``"iris"``, ``"mpg"``).
    difficulty : str | Difficulty
        ``"easy"``, ``"medium"``, or ``"hard"``.
    problem_type : str | ProblemType | None
        ``"classification"`` or ``"regression"``. Auto-detected if *None*.
    metric : str | None
        Primary metric. Defaults to ``"accuracy"`` (classification)
        or ``"r2"`` (regression).
    max_steps : int | None
        Maximum ``RunCode`` executions per episode.
    timeout_seconds : float
        Per-execution wall-clock time limit.
    seed : int
        RNG seed for reproducibility.

    Example
    -------
    >>> env = ODSEnvironment(dataset="titanic", difficulty="easy")
    >>> obs = env.reset()
    >>> result = env.step(RunCodeAction(code="print(train_df.shape)"))
    >>> result = env.step(RunCodeAction(code=\"\"\"
    ... from sklearn.linear_model import LogisticRegression
    ... from sklearn.preprocessing import LabelEncoder
    ... X = train_df.drop('survived', axis=1).select_dtypes('number').fillna(0)
    ... y = train_df['survived']
    ... model = LogisticRegression(max_iter=1000).fit(X, y)
    ... predictions = model.predict(test_features.select_dtypes('number').fillna(0))
    ... \"\"\"))
    >>> result = env.step(SubmitAction())
    >>> print(result.info["test_report"])
    """

    MAX_STEPS: Dict[Difficulty, int] = {
        Difficulty.EASY: 20,
        Difficulty.MEDIUM: 40,
        Difficulty.HARD: 60,
    }

    def __init__(
        self,
        dataset: str = "titanic",
        difficulty: Difficulty | str = Difficulty.EASY,
        problem_type: ProblemType | str | None = None,
        metric: str | None = None,
        max_steps: int | None = None,
        timeout_seconds: float = 30.0,
        seed: int = 42,
    ) -> None:
        if isinstance(difficulty, str):
            difficulty = Difficulty(difficulty)

        self.dataset_name = dataset
        self.difficulty = difficulty
        self.seed = seed
        self.timeout_seconds = timeout_seconds

        # Load dataset config
        self._dataset_config: DatasetConfig = load_dataset(dataset, difficulty)

        # Determine problem type
        if problem_type is not None:
            self.problem_type = (
                ProblemType(problem_type)
                if isinstance(problem_type, str)
                else problem_type
            )
        else:
            self.problem_type = self._dataset_config.problem_type

        # Determine metric
        if metric is not None:
            self.metric = metric
        else:
            self.metric = (
                "accuracy"
                if self.problem_type == ProblemType.CLASSIFICATION
                else "r2"
            )

        # Max steps
        self.max_steps = max_steps or self.MAX_STEPS.get(difficulty, 30)

        # Internal state (populated on reset)
        self._data_split: Optional[DataSplit] = None
        self._executor: Optional[SandboxExecutor] = None
        self._step_count: int = 0
        self._done: bool = False
        self._best_val_score: Optional[float] = None
        self._prev_val_score: Optional[float] = None
        self._had_predictions: bool = False

    # ========================================================================
    # Public API
    # ========================================================================

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        np.random.seed(self.seed)

        # Create train / val / test split
        self._data_split = create_data_split(
            self._dataset_config,
            seed=self.seed,
        )

        # Set up sandbox executor
        self._executor = SandboxExecutor(
            timeout_seconds=self.timeout_seconds,
        )
        self._executor.setup_namespace(
            train_df=self._data_split.train_df,
            val_features=self._data_split.val_features,
            test_features=self._data_split.test_features,
            target_column=self._dataset_config.target_column,
            evaluate_fn=self._make_evaluate_fn(),
        )

        # Reset episode counters
        self._step_count = 0
        self._done = False
        self._best_val_score = None
        self._prev_val_score = None
        self._had_predictions = False

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute *action* and return a :class:`StepResult`."""
        self._ensure_ready()

        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new one."
            )

        self._step_count += 1

        if isinstance(action, SubmitAction):
            return self._handle_submit()
        if isinstance(action, RunCodeAction):
            return self._handle_run_code(action)

        raise ValueError(f"Unknown action type: {type(action)}")

    def state(self) -> Observation:
        """Return current observation without advancing the episode."""
        self._ensure_ready()
        return self._build_observation()

    # ========================================================================
    # Submit handler
    # ========================================================================

    def _handle_submit(self) -> StepResult:
        """Score predictions on hidden test set and terminate."""
        self._done = True

        predictions = self._executor.get_predictions()
        test_score: Optional[float] = None
        test_report: Dict[str, Any] = {}

        if (
            predictions is not None
            and len(predictions) == len(self._data_split.test_labels)
        ):
            try:
                test_score = compute_metric(
                    self._data_split.test_labels.values,
                    predictions,
                    self.problem_type,
                    self.metric,
                )
                test_report = compute_full_report(
                    self._data_split.test_labels.values,
                    predictions,
                    self.problem_type,
                )
            except Exception as exc:
                test_report = {"error": str(exc)}

        reward = compute_submit_reward(
            test_score=test_score,
            best_validation_score=self._best_val_score,
        )

        obs = self._build_observation(done=True)

        return StepResult(
            observation=obs,
            reward=reward,
            done=True,
            info={
                "reason": "submit",
                "test_score": test_score,
                "test_report": test_report,
                "best_validation_score": self._best_val_score,
            }
        )

    # ========================================================================
    # RunCode handler
    # ========================================================================

    def _handle_run_code(self, action: RunCodeAction) -> StepResult:
        """Execute code in the sandbox and compute the dense reward."""
        had_preds = self._had_predictions
        prev_val = self._prev_val_score

        # Execute the agent's code
        result = self._executor.execute(action.code)
        code_ok = result.status == ExecutionStatus.SUCCESS

        # Check for predictions variable
        predictions = self._executor.get_predictions()
        has_preds = predictions is not None
        if has_preds:
            self._had_predictions = True

        # Auto-score against validation if predictions match val size
        curr_val_score: Optional[float] = None
        if (
            has_preds
            and len(predictions) == len(self._data_split.val_labels)
        ):
            try:
                curr_val_score = compute_metric(
                    self._data_split.val_labels.values,
                    predictions,
                    self.problem_type,
                    self.metric,
                )
                if (
                    self._best_val_score is None
                    or curr_val_score > self._best_val_score
                ):
                    self._best_val_score = curr_val_score
            except Exception:
                curr_val_score = None

        self._prev_val_score = curr_val_score

        # Dense reward
        reward = compute_step_reward(
            code_succeeded=code_ok,
            had_predictions_before=had_preds,
            has_predictions_now=has_preds,
            prev_validation_score=prev_val,
            curr_validation_score=curr_val_score,
        )

        # Check step-budget termination
        done = self._step_count >= self.max_steps
        if done:
            self._done = True

        obs = self._build_observation(
            stdout=result.stdout,
            stderr=result.stderr,
            execution_status=result.status,
            execution_time_ms=result.execution_time_ms,
            validation_score=curr_val_score,
            done=done,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "execution_status": result.status.value,
                "validation_score": curr_val_score,
                "best_validation_score": self._best_val_score,
            },
        )

    # ========================================================================
    # Observation builder
    # ========================================================================

    def _build_observation(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        execution_status: Optional[ExecutionStatus] = None,
        execution_time_ms: float = 0.0,
        validation_score: Optional[float] = None,
        done: bool = False,
    ) -> Observation:
        return Observation(
            stdout=stdout,
            stderr=stderr,
            execution_status=execution_status,
            execution_time_ms=execution_time_ms,
            namespace_summary=self._executor.get_namespace_summary(),
            validation_score=validation_score,
            best_validation_score=self._best_val_score,
            step_count=self._step_count,
            max_steps=self.max_steps,
            dataset_info=self._build_dataset_info(),
            task_description=self._build_task_description(),
            done=done,
        )

    def _build_dataset_info(self) -> DatasetInfo:
        split = self._data_split
        train = split.train_df
        cfg = self._dataset_config

        columns = []
        for col in train.columns:
            columns.append(
                ColumnSchema(
                    name=col,
                    dtype=str(train[col].dtype),
                    null_count=int(train[col].isnull().sum()),
                    is_numeric=pd.api.types.is_numeric_dtype(train[col]),
                    unique_count=int(train[col].nunique()),
                    sample_values=train[col].dropna().head(3).tolist(),
                )
            )

        info = DatasetInfo(
            train_shape=tuple(train.shape),
            val_shape=tuple(split.val_features.shape),
            test_shape=tuple(split.test_features.shape),
            target_column=cfg.target_column,
            problem_type=self.problem_type.value,
            metric=self.metric,
            columns=columns,
        )

        # Target-specific metadata
        target_col = train[cfg.target_column]
        if self.problem_type == ProblemType.CLASSIFICATION:
            info.target_classes = sorted(
                target_col.dropna().unique().tolist(),
                key=str,
            )
        else:
            info.target_stats = {
                "mean": round(float(target_col.mean()), 4),
                "std": round(float(target_col.std()), 4),
                "min": round(float(target_col.min()), 4),
                "max": round(float(target_col.max()), 4),
            }

        return info

    def _build_task_description(self) -> str:
        pt = self.problem_type.value.upper()
        tc = self._dataset_config.target_column
        return (
            f"{pt} TASK: Build a model to predict '{tc}' using the "
            f"provided training data.\n\n"
            f"Your code runs in a persistent sandbox - variables survive "
            f"across RunCode steps.\n"
            f"Pre-loaded variables: train_df, val_features, test_features, "
            f"target_column.\n"
            f"Use evaluate(predictions) to check your score on hidden "
            f"validation labels (pass val-sized predictions).\n"
            f"When ready, set `predictions` to your test-set predictions "
            f"(matching test_features length) and call Submit.\n\n"
            f"Primary metric: {self.metric} | "
            f"Max steps: {self.max_steps} | "
            f"Dataset: {self.dataset_name} ({self.difficulty.value})"
        )

    # ========================================================================
    # evaluate() factory
    # ========================================================================

    def _make_evaluate_fn(self):
        """Create the ``evaluate()`` closure injected into the namespace.

        Scores predictions against the **hidden validation labels**.
        Also updates ``self._best_val_score`` as a side-effect so the
        environment can track progress for reward shaping.
        """
        val_labels = self._data_split.val_labels.copy()
        problem_type = self.problem_type
        metric = self.metric
        env = self  # capture for side-effect updates

        def evaluate(predictions) -> Dict[str, Any]:
            """Score predictions against hidden validation labels.

            Parameters
            ----------
            predictions : array-like
                Predictions whose length must match ``val_features``.

            Returns
            -------
            dict
                Metric scores plus ``primary_metric`` and ``primary_score``.
            """
            preds = np.asarray(predictions)
            if len(preds) != len(val_labels):
                return {
                    "error": (
                        f"Expected {len(val_labels)} predictions "
                        f"(val_features length), got {len(preds)}"
                    )
                }

            try:
                primary = compute_metric(
                    val_labels.values, preds, problem_type, metric,
                )
                report = compute_full_report(
                    val_labels.values, preds, problem_type,
                )
                report["primary_metric"] = metric
                report["primary_score"] = round(primary, 4)

                # Side-effect: let the environment know about the score
                if (
                    env._best_val_score is None
                    or primary > env._best_val_score
                ):
                    env._best_val_score = primary

                return report
            except Exception as exc:
                return {"error": str(exc)}

        return evaluate

    # ========================================================================
    # Internals
    # ========================================================================

    def _ensure_ready(self) -> None:
        if self._executor is None or self._data_split is None:
            raise RuntimeError(
                "Environment not initialised - call reset() first."
            )