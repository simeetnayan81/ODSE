"""Example usage of the ODSE environment.

Run with:  python -m core.example   (from the ODSE root directory)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ODSE package root is on sys.path when running as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.env import ODSEnvironment, grade_performance
from core.models import (
    CreateInteractionAction,
    Difficulty,
    DropColumnAction,
    ImputeAction,
    ScaleColumnAction,
    SubmitAction,
    TaskType,
)


def run_cleaning_example() -> None:
    """Run a data-cleaning episode on the easy Titanic dataset."""
    print("=" * 60)
    print("  DATA CLEANING EXAMPLE  (easy)")
    print("=" * 60)

    env = ODSEnvironment(
        task_type=TaskType.DATA_CLEANING,
        difficulty=Difficulty.EASY,
        seed=42,
    )
    obs = env.reset()
    print(f"\nInitial state:")
    print(f"  Shape:             {obs.shape}")
    print(f"  Nulls remaining:   {obs.nulls_remaining}")
    print(f"  Accuracy:          {obs.current_accuracy:.4f}")
    print(f"  Goal:              {obs.goal_description}")
    print(f"  Actions:           {obs.available_actions}")
    print()

    # Step 1 - impute Age with median
    result = env.step(ImputeAction(column="age", strategy="median"))
    print(
        f"Step 1  impute(age, median)      ->  "
        f"reward={result.reward:+.4f}  "
        f"nulls={result.observation.nulls_remaining}  "
        f"acc={result.observation.current_accuracy:.4f}"
    )

    # Step 2 - drop Cabin (many nulls)
    result = env.step(DropColumnAction(column="cabin"))
    print(
        f"Step 2  drop_column(cabin)       ->  "
        f"reward={result.reward:+.4f}  "
        f"nulls={result.observation.nulls_remaining}  "
        f"acc={result.observation.current_accuracy:.4f}"
    )

    # Step 3 - impute Embarked with mode
    result = env.step(ImputeAction(column="embarked", strategy="mode"))
    print(
        f"Step 3  impute(embarked, mode)   ->  "
        f"reward={result.reward:+.4f}  "
        f"nulls={result.observation.nulls_remaining}  "
        f"acc={result.observation.current_accuracy:.4f}"
    )

    # Submit
    result = env.step(SubmitAction())
    print(f"\nSubmitted!  done={result.done}")

    # Grade
    report = grade_performance(env)
    print("\n--- Grading Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

def run_feature_engineering_example() -> None:
    """Run a feature-engineering episode on the easy Iris dataset."""
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING EXAMPLE  (easy)")
    print("=" * 60)

    env = ODSEnvironment(
        task_type=TaskType.FEATURE_ENGINEERING,
        difficulty=Difficulty.EASY,
        seed=42,
    )

    obs = env.reset()
    print(f"\nInitial state:")
    print(f"  Shape:             {obs.shape}")
    print(f"  Accuracy:          {obs.current_accuracy:.4f}")
    print(f"  Goal:              {obs.goal_description}")
    print(f"  Actions:           {obs.available_actions}")
    print()

    # Step 1 - Scale a numerical column
    result = env.step(ScaleColumnAction(column="sepal_length", method="standard"))
    print(
        f"Step 1  scale(sepal_length, standard)        ->  "
        f"reward={result.reward:+.4f}  "
        f"acc={result.observation.current_accuracy:.4f}"
    )

    # Step 2 - Create an interaction feature
    result = env.step(CreateInteractionAction(
        column_a="petal_length", 
        column_b="petal_width", 
        new_column="petal_area"
    ))
    print(
        f"Step 2  interact(petal_length * petal_width) ->  "
        f"reward={result.reward:+.4f}  "
        f"acc={result.observation.current_accuracy:.4f}"
    )

    # Submit
    result = env.step(SubmitAction())
    print(f"\nSubmitted!  done={result.done}")

    # Grade
    report = grade_performance(env)
    print("\n--- Grading Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_cleaning_example()
    run_feature_engineering_example()