from abc import ABC, abstractmethod
from typing import Tuple, Any
from ..data.data_manager import DataState

class BaseTask(ABC):
    def __init__(self, task_id: str, difficulty: str, dataset_path: str):
        self.task_id = task_id
        self.difficulty = difficulty
        self.dataset_path = dataset_path
        
    @abstractmethod
    def get_initial_state(self) -> DataState:
        """Returns the starting data state for the task."""
        pass

    @abstractmethod
    def calculate_reward(self, old_df: pd.DataFrame, new_df: pd.DataFrame, action: Any) -> float:
        """
        Logic for rewarding the agent. 
        Example: Reward = (Accuracy_new - Accuracy_old) - Penalty
        """
        pass

    @abstractmethod
    def is_done(self, current_df: pd.DataFrame, step_count: int) -> bool:
        """Win/Loss conditions (e.g., all nulls gone or max steps reached)."""
        pass

    @abstractmethod
    def get_goal_description(self) -> str:
        """Human-readable goal for the agent's prompt."""
        pass