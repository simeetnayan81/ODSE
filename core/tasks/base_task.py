from abc import ABC, abstractmethod
from core.data_manager import DataState

class BaseTask(ABC):
    def __init__(self, task_id: str, difficulty: str, dataset_path: str):
        self.task_id = task_id
        self.difficulty = difficulty
        self.dataset_path = dataset_path

    @abstractmethod
    def get_initial_state(self) -> DataState:
        """Returns the initial wrapped DataState."""
        pass

    @abstractmethod
    def calculate_reward(self, old_state: DataState, new_state: DataState, action: Any) -> float:
        """Compare two DataState objects to find the delta."""
        pass

    @abstractmethod
    def is_done(self, current_state: DataState, step_count: int) -> bool:
        pass