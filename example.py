"""Example usage of ODSE environment."""

from env import ODSEEnvironment, grade_performance
from models import ImputeAction, DropAction, SubmitAction


def main():
    """Run a simple example episode."""
    
    # Initialize environment
    env = ODSEEnvironment(seed=42)
    
    # Reset and get initial observation
    obs = env.reset()
    print("=== Initial State ===")
    print(f"Nulls remaining: {obs.nulls_remaining}")
    print(f"Current accuracy: {obs.current_accuracy:.4f}")
    print(f"Column metadata: {obs.column_metadata}\n")
    
    # Example: Impute Age with mean
    print("=== Step 1: Impute Age with mean ===")
    action = ImputeAction(column="Age", strategy="mean")
    result = env.step(action)
    print(f"Reward: {result.reward:.4f}")
    print(f"Nulls remaining: {result.observation.nulls_remaining}")
    print(f"Current accuracy: {result.observation.current_accuracy:.4f}")
    print(f"Info: {result.info}\n")
    
    # Example: Drop Cabin (has many nulls)
    print("=== Step 2: Drop Cabin column ===")
    action = DropAction(column="Cabin")
    result = env.step(action)
    print(f"Reward: {result.reward:.4f}")
    print(f"Nulls remaining: {result.observation.nulls_remaining}")
    print(f"Current accuracy: {result.observation.current_accuracy:.4f}\n")
    
    # Example: Impute Embarked with mode
    print("=== Step 3: Impute Embarked with mode ===")
    action = ImputeAction(column="Embarked", strategy="mode")
    result = env.step(action)
    print(f"Reward: {result.reward:.4f}")
    print(f"Nulls remaining: {result.observation.nulls_remaining}")
    print(f"Current accuracy: {result.observation.current_accuracy:.4f}\n")
    
    # Submit the episode
    print("=== Step 4: Submit ===")
    action = SubmitAction()
    result = env.step(action)
    print(f"Episode done: {result.done}")
    print(f"Final nulls remaining: {result.observation.nulls_remaining}")
    print(f"Final accuracy: {result.observation.current_accuracy:.4f}\n")
    
    # Grade the performance
    print("=== Grading ===")
    score = grade_performance(env.working_df, env)
    print(f"Performance score: {score:.2f}/1.0")


if __name__ == "__main__":
    main()
