# graders.py

SUCCESS_SCORE_THRESHOLD_EASY = 0.5
SUCCESS_SCORE_THRESHOLD_MEDIUM = 0.75
SUCCESS_SCORE_THRESHOLD_HARD = 0.9

class BaseGrader:
    def clamp_score(self, raw_score: float) -> float:
        """Strictly bound the score for the Hackathon server."""
        return max(0.01, min(0.99, float(raw_score)))

    def get_score(self, obs) -> float:
        """Compute the final score from the observation."""
        raw_score = 0.0
        try:
            if hasattr(obs, "stderr") and obs.stderr and obs.stderr.strip():
                # If there's an error during code execution, penalize the score
                raw_score = 0.0
            elif hasattr(obs, "test_score") and obs.test_score is not None:
                raw_score = float(obs.test_score)
            elif hasattr(obs, "best_validation_score") and obs.best_validation_score is not None:
                raw_score = float(obs.best_validation_score)
            elif hasattr(obs, "validation_score") and obs.validation_score is not None:
                raw_score = float(obs.validation_score)
            else:
                raw_score = 0.0
        except Exception:
            raw_score = 0.0
            
        return self.clamp_score(raw_score)

    def __call__(self, obs, **kwargs) -> float:
        """Makes the class instance callable like a function."""
        return self.get_score(obs)

class EasyGrader(BaseGrader):
    """Callable class for grading the easy task."""
    pass

class MediumGrader(BaseGrader):
    """Callable class for grading the medium task."""
    pass

class HardGrader(BaseGrader):
    """Callable class for grading the hard task."""
    pass