"""Reward shaping for the ODSE sandbox environment.

Two kinds of reward:

1. **Dense (heuristic)**: computed every ``RunCode`` step to give the
   agent a gradient signal. Components:
   - Step penalty (discourages dawdling)
   - Code success / error bonus / penalty
   - First-time predictions bonus
   - Validation-score improvement (proportional to delta)

2. **Sparse (final)**: computed once on ``Submit``; this is the true
   test-set score that the agent is ultimately optimising.
"""

from __future__ import annotations

from typing import Optional

# -- Hyperparameters (tune as needed) ----------------------------------------

STEP_PENALTY: float = -0.01
CODE_SUCCESS_BONUS: float = 0.05
CODE_ERROR_PENALTY: float = -0.05
FIRST_PREDICTION_BONUS: float = 0.1
VALIDATION_IMPROVEMENT_SCALE: float = 5.0

# ============================================================================
# Dense reward (per RunCode step)
# ============================================================================

def compute_step_reward(
    *,
    code_succeeded: bool,
    had_predictions_before: bool,
    has_predictions_now: bool,
    prev_validation_score: Optional[float],
    curr_validation_score: Optional[float],
) -> float:
    """Return the dense heuristic reward for one ``RunCode`` step.

    Parameters
    ----------
    code_succeeded
        Whether the code ran without errors.
    had_predictions_before
        Whether a ``predictions`` variable existed before this step.
    has_predictions_now
        Whether a ``predictions`` variable exists after this step.
    prev_validation_score
        Validation score *before* this step (or ``None``).
    curr_validation_score
        Validation score *after* this step (or ``None``).

    Returns
    -------
    float
        Scalar reward (can be negative).
    """
    reward = STEP_PENALTY

    # Code execution outcome
    reward += CODE_SUCCESS_BONUS if code_succeeded else CODE_ERROR_PENALTY

    # First time producing predictions
    if has_predictions_now and not had_predictions_before:
        reward += FIRST_PREDICTION_BONUS


    # Should we penalize if current predictions are worse than before? 
    # Maybe not, since some steps may be exploratory and temporarily break the pipeline.
    # Should we penalize for breaking existing predictions? Maybe not, since some feature engineering steps may temporarily break the pipeline.
    # Let's just give a small step penalty and let the validation score guide the agent back on track.
    # Validation score improvement
    if curr_validation_score is not None and prev_validation_score is not None:
        delta = curr_validation_score - prev_validation_score
        if delta > 0:
            reward += delta * VALIDATION_IMPROVEMENT_SCALE

    return reward

# ============================================================================
# Sparse reward (on Submit)
# ============================================================================

def compute_submit_reward(
    *,
    test_score: Optional[float],
    best_validation_score: Optional[float],
) -> float:
    """Return the final reward when the agent calls ``Submit``.

    Parameters
    ----------
    test_score
        Metric score on the hidden test set (higher-is-better).
        ``None`` means the agent failed to produce valid predictions.
    best_validation_score
        Best validation score achieved during the episode (informational).

    Returns
    -------
    float
        The test-set score (0.0 if no valid predictions were produced).
    """
    if test_score is None:
        return 0.0
    return float(test_score)