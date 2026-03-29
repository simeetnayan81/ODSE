# ODSE
Open Data Science Environment (ODSE): A standardized environment for AI agents to master end-to-end data science pipelines.

## Overview

ODSE is a Reinforcement Learning environment built for Data Science Sandbox. It simulates the various **Data Science** task where agents learn to handle missing values and improve model accuracy. The environment follows a strict API specification: `reset()`, `state()`, and `step(action)`.

## Features

- **Pydantic-typed API**: Fully type-safe observations and actions using Pydantic v2 discriminated unions
- **Pandas-based State Management**: Internal state managed as a working DataFrame
- **Scikit-learn Integration**: Fast accuracy evaluation via 5-fold cross-validation with LogisticRegression
- **Embedded Titanic Dataset**: Pre-packaged dirty Titanic CSV for quick testing
- **Structured Rewards**: Accuracy-based rewards with step penalties
- **Performance Grading**: Built-in grader function for evaluating agent performance


## Installation

```bash
pip install -r requirements.txt
```

## API Reference

### Environment Interface

#### `reset() → Observation`
Loads a dirty Titanic dataset and resets the episode. Returns initial observation.

**Example:**
```python
env = ODSEEnvironment(seed=42)
obs = env.reset()
```

#### `state() → Observation`
Returns current environment state with column metadata and model accuracy.

**Observation fields:**
- `column_metadata`: Dict mapping column names to `{null_count, type, null_percentage}`
- `sample_head`: First 5 rows as JSON dict
- `current_accuracy`: 5-fold CV accuracy on LogisticRegression (0.0-1.0)
- `step_count`: Number of actions taken
- `nulls_remaining`: Total missing values in dataset

#### `step(action: Action) → StepResult`
Executes an action and returns the result.

**Action Types (Discriminated Union):**

1. **ImputeAction**
   ```python
   ImputeAction(column="Age", strategy="mean" | "median" | "mode")
   ```
   Fills missing values using the specified strategy.

2. **DropAction**
   ```python
   DropAction(column="Cabin")
   ```
   Removes a column entirely.

3. **SubmitAction**
   ```python
   SubmitAction()
   ```
   Terminates the episode and triggers final evaluation.

**StepResult fields:**
- `observation`: Updated observation
- `reward`: (accuracy_gain × 10.0) - 0.01 (step penalty)
- `done`: True if episode terminated
- `info`: Dict with debugging info

#### `grade_performance(final_df: DataFrame) → float`
Evaluates final performance on a 0.0-1.0 scale.


## Quick Start

```python
from env import ODSEEnvironment, grade_performance
from models import ImputeAction, DropAction, SubmitAction

# Initialize
env = ODSEEnvironment(seed=42)
obs = env.reset()

# Take actions
action = ImputeAction(column="Age", strategy="mean")
result = env.step(action)

action = DropAction(column="Cabin")
result = env.step(action)

# Submit episode
result = env.step(SubmitAction())
score = grade_performance(env.working_df)
print(f"Final Score: {score:.2f}")
```

## Example Run

Run the example script:
```bash
python example.py
```

## Reward Structure

The agent receives rewards based on:
```
reward = (new_accuracy - old_accuracy) × 10 - 0.01
```

- **Positive reward**: Accuracy improves (incentivizes data quality)
- **Step penalty**: -0.01 per action (encourages efficiency)
- **Submit bonus**: No penalty for submitting

## Termination Conditions

An episode ends when:
1. Agent calls `SubmitAction()`
2. All missing values are imputed/removed (`nulls_remaining == 0`)

## Implementation Notes

- **Dataset**: Dirty Titanic CSV with 20 rows and ~20% missing values in key features
- **Target**: Survived (binary classification)
- **Features**: Auto-selects numeric and categorical columns (excludes PassengerId, Name, Ticket, Cabin from features by default in grading)
- **CV Strategy**: 5-fold cross-validation for robust accuracy estimates
- **Encoding**: LabelEncoder for categorical features during model training

## Dependencies

- `pydantic>=2.0.0`: Type validation and serialization
- `pandas>=1.5.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning models
- `numpy>=1.24.0`: Numerical computing
- `typing-extensions>=4.5.0`: Advanced type hints

---