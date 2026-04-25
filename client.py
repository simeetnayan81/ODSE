"""ODSE Environment Client.

Provides the ``OdseEnv`` class which connects to a running ODSE server
(or auto-launches one from a Docker image) via WebSocket.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import OdseAction, OdseObservation, OdseState


class OdseEnv(EnvClient[OdseAction, OdseObservation, OdseState]):
    """Client for the ODSE environment.

    Example (connect to running server)::

        with OdseEnv(base_url="http://localhost:8000") as client:
            result = client.reset()
            print(result.observation.task_description)

            result = client.step(
                OdseAction(action_type="run_code", code="print(train_df.shape)")
            )
            print(result.observation.stdout)

            result = client.step(OdseAction(action_type="submit"))
            print(result.reward)

    Example (Docker)::

        client = OdseEnv.from_docker_image("odse-env:latest")
        try:
            result = client.reset()
            result = client.step(
                OdseAction(action_type="run_code", code="print(train_df.head())")
            )
        finally:
            client.close()
    """

    def _step_payload(self, action: OdseAction) -> Dict[str, Any]:
        """Convert ``OdseAction`` to JSON payload for the step message."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.code is not None:
            payload["code"] = action.code
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OdseObservation]:
        """Parse server response into ``StepResult[OdseObservation]``."""
        obs_data = payload.get("observation", {})
        # serialize_observation strips done/reward from the obs dict and sends
        # them at the top level - stamp them back so the observation is complete.
        obs_data["done"] = payload.get("done", False)
        obs_data["reward"] = payload.get("reward")
        observation = OdseObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OdseState:
        """Parse server response into ``OdseState``."""
        return OdseState.model_validate(payload)