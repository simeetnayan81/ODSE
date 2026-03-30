from .base_task import BaseTask
from ..data.data_manager import DataState

class CleaningTask(BaseTask):
    def get_initial_state(self) -> DataState:
        return DataState(self.initial_df, name=f"{self.task_id}_start")

    def calculate_reward(self, old_state: DataState, new_state: DataState, action: Any) -> float:
        """
        Reward is calculated based on the reduction of null values 
        stored in the DataState metadata.
        """
        # O(1) access to pre-calculated null counts
        old_nulls = sum(old_state.metadata["null_counts"].values())
        new_nulls = sum(new_state.metadata["null_counts"].values())
        
        # Improvement Reward
        # R = (Nulls_dropped * weight) - Step_penalty
        null_reduction = old_nulls - new_nulls
        reward = null_reduction * 0.1
        
        # Constant penalty per step to discourage 'brute force' or 
        # repeating the same action.
        step_penalty = 0.01
        
        return reward - step_penalty

    def is_done(self, current_state: DataState, step_count: int) -> bool:
        """
        Termination logic using metadata.
        """
        total_nulls = sum(current_state.metadata["null_counts"].values())
        
        # Win: No missing values remain
        # Lose: Step limit reached (prevent infinite loops)
        max_steps = 15
        return total_nulls == 0 or step_count >= max_steps

    def get_goal_description(self) -> str:
        return (
            "CLEANING TASK: Your objective is to reach a zero-null state. "
            "Use imputers or row-dropping actions effectively. "
            "Fewer steps result in a higher final score."
        )