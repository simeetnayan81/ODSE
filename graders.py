# graders.py

SUCCESS_SCORE_THRESHOLD_EASY = 0.5
SUCCESS_SCORE_THRESHOLD_MEDIUM = 0.75
SUCCESS_SCORE_THRESHOLD_HARD = 0.9

class BaseGrader:
    def clamp_score(self, raw_score: float) -> float:
        """Strictly bound the score for the Hackathon server."""
        return max(0.01, min(0.99, float(raw_score)))

    def grade(self, obs, **kwargs) -> float:
        """Compute a score based on the observation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the grade() method.")
    
    def get_score(self, obs, **kwargs) -> float:
        """Public method to compute the final score."""
        raw_score = 0.0
        try:
            if obs.stderr and obs.stderr.strip():
                # If there's an error during code execution, we can choose to penalize the score
                raw_score = 0.0
            elif "test_score" in obs:
                raw_score = obs.get("test_score", 0.0)
            elif "best_validation_score" in obs:
                raw_score = obs.get("best_validation_score", 0.0)
            elif "validation_score" in obs:
                raw_score = obs.get("validation_score", 0.0)
            else:
                raw_score = 0.0
        except Exception:
            raw_score = 0.0
        return self.clamp_score(raw_score)

class EasyGrader(BaseGrader):
    def grade(self, obs, **kwargs) -> float:
        score = self.get_score(obs)
        return score

class MediumGrader(BaseGrader):
    def grade(self, obs, **kwargs) -> float:
        score = self.get_score(obs)
        return score

class HardGrader(BaseGrader):
    def grade(self, obs, **kwargs) -> float:
        score = self.get_score(obs)
        return score