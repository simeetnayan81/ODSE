"""
ODSE Environment Implementation (OpenEnv-compatible).

Wraps the core ``ODSEnvironment`` in the OpenEnv ``Environment`` interface
so it can be served over HTTP / WebSocket via the standard OpenEnv server.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from odse.models import OdseAction, OdseObservation, OdseState
from odse.core.env import ODSEnvironment
from odse.core.models import RunCodeAction, SubmitAction


class OdseEnvironment(Environment):
    """OpenEnv wrapper around the core ODSE sandbox environment.

    Each episode presents the agent with a dataset, a sandbox executor,
    and asks it to build a predictive model by writing Python code.

    Example::

        env = OdseEnvironment()
        obs = env.reset()
        print(obs.task_description)
        obs = env.step(OdseAction(action_type="run_code",
                                  code="print(train_df.shape)"))
        print(obs.stdout)
        obs = env.step(OdseAction(action_type="submit"))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # Docker executor is not concurrency-safe

    def __init__(
        self,
        dataset: str = "breast_cancer",
        difficulty: str = "easy",
        problem_type: Optional[str] = None,
        metric: Optional[str] = None,
        max_steps: Optional[int] = None,
        timeout_seconds: float = 30.0,
        seed: int = 42,
    ):
        super().__init__()
        self._dataset = dataset
        self._difficulty = difficulty
        self._problem_type = problem_type
        self._metric = metric
        self._max_steps = max_steps
        self._timeout_seconds = timeout_seconds
        self._seed = seed

        self._core_env: Optional[ODSEnvironment] = None
        self._state = OdseState(episode_id=str(uuid4()), step_count=0)

    # ---------------------------------------------------------------------- reset
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OdseObservation:
        """Reset the environment and return the initial observation."""

        effective_seed = seed if seed is not None else self._seed

        # Allow per-episode overrides via kwargs
        dataset = kwargs.get("dataset", self._dataset)
        difficulty = kwargs.get("difficulty", self._difficulty)
        p_type = kwargs.get("problem_type", self._problem_type)
        metric = kwargs.get("metric", self._metric)
        max_steps = kwargs.get("max_steps", self._max_steps)

        self._core_env = ODSEnvironment(
            dataset=dataset,
            difficulty=difficulty,
            problem_type=p_type,
            metric=metric,
            max_steps=max_steps,
            timeout_seconds=self._timeout_seconds,
            seed=effective_seed,
        )

        core_obs = self._core_env.reset()

        # Build state
        self._state = OdseState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            dataset_name=self._core_env.dataset_name,
            difficulty=self._core_env.difficulty.value,
            problem_type=self._core_env.problem_type.value,
            target_column=self._core_env.dataset_config.target_column,
            metric=self._core_env.metric,
            max_steps=self._core_env.max_steps,
            done=False,
            best_validation_score=None,
            latest_validation_score=None,
        )

        return OdseObservation(**core_obs.model_dump(), reward=0.0)

    # ---------------------------------------------------------------------- step
    def step(
        self,
        action: OdseAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OdseObservation:
        """Execute an action and return the resulting observation."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        # Convert OpenEnv action -> core action
        if action.action_type == "run_code":
            if not action.code:
                raise ValueError(
                    "'action_type='run_code' requires a non-empty 'code' field."
                )
            core_action = RunCodeAction(code=action.code)
        elif action.action_type == "submit":
            core_action = SubmitAction()
        else:
            raise ValueError(f"Unknown action_type: {action.action_type!r}")

        step_result = self._core_env.step(core_action)

        # Sync state
        self._state.step_count = step_result.observation.step_count
        self._state.done = step_result.done
        self._state.best_validation_score = step_result.observation.best_validation_score
        self._state.latest_validation_score = step_result.observation.validation_score

        # Construct OpenEnv observation from core observation
        return OdseObservation(
            **step_result.observation.model_dump(),
            reward=step_result.reward,
            info=step_result.info,
        )

    # ---------------------------------------------------------------------- state
    @property
    def state(self) -> OdseState:
        """Current environment state."""
        return self._state
