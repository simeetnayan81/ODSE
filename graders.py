# graders.py

SUCCESS_SCORE_THRESHOLD_EASY = 0.5
SUCCESS_SCORE_THRESHOLD_MEDIUM = 0.75
SUCCESS_SCORE_THRESHOLD_HARD = 0.9

def clamp_score(raw_score: float) -> float:
    """Strictly bound the score for the Hackathon server."""
    return max(0.01, min(0.99, float(raw_score)))

def get_score(obs) -> float:
    """Compute the final score from the observation."""
    raw_score = 0.0
    try:
        if hasattr(obs, "stderr") and obs.stderr and obs.stderr.strip():
            # If there's an error during code execution, we can choose to penalize the score
            raw_score = 0.0
        elif hasattr(obs, "test_score") and obs.test_score is not None:
            raw_score = obs.test_score
        elif hasattr(obs, "best_validation_score") and obs.best_validation_score is not None:
            raw_score = obs.best_validation_score
        elif hasattr(obs, "validation_score") and obs.validation_score is not None:
            raw_score = obs.validation_score
        else:
            raw_score = 0.0
    except Exception:
        raw_score = 0.0
    return clamp_score(raw_score)

def grade_easy(obs) -> float:
    """Grade the easy task."""
    return get_score(obs)

def grade_medium(obs) -> float:
    """Grade the medium task."""
    return get_score(obs)

def grade_hard(obs) -> float:
    """Grade the hard task."""
    return get_score(obs)