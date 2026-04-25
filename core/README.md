# Open Data Science Sandbox Environment (ODSE)

ODSE is a persistent, sandboxed Reinforcement Learning (RL) environment designed to train and evaluate autonomous AI agents on data science tasks. 

Instead of choosing from a predefined list of discrete actions, agents interact with ODSE by writing and executing raw Python code. The environment evaluates the agent's ability to explore data, handle missing values, train machine learning models, and submit predictions on hidden test sets.

## Key Features

* **Notebook-Style Execution:** Agents execute arbitrary Python code in a persistent namespace. Variables, models, and data frames survive across execution steps.
* **Built-in Datasets:** Includes classification and regression tasks (Breast Cancer, Iris, Wine, House Prices) with configurable difficulty levels (Easy, Medium, Hard) that inject noise and missing values.
* **Safe Sandboxing:** Code is executed with strict wall-clock timeouts, stdout/stderr truncation, and restricted imports (whitelisting `pandas`, `sklearn`, `numpy`, etc.) to prevent environment corruption.
* **Dense Reward Shaping:** Designed for RL. Agents receive immediate heuristic rewards for successful code execution, generating predictions, and improving validation scores, mitigating the sparse-reward problem in code generation.
* **Pydantic Data Contracts:** Strongly typed actions, observations, and step results.

## Quick Start

```python
from core.env import ODSEnvironment
from core.models import RunCodeAction, SubmitAction

# 1. Initialize the environment
env = ODSEnvironment(dataset="breast_cancer", difficulty="easy")
obs = env.reset()

print(obs.task_description)

# 2. Agent explores the data
result = env.step(RunCodeAction(code="""
print(train_df.head())
print(train_df.isnull().sum())
"""))
print(result.observation.stdout)

# 3. Agent trains a model
result = env.step(RunCodeAction(code="""
from sklearn.linear_model import LogisticRegression
X = train_df.drop('target', axis=1)
y = train_df['target']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Score on validation set
val_preds = model.predict(val_features)
print(evaluate(val_preds))
"""))

# 4. Agent predicts on the test set and submits
result = env.step(RunCodeAction(code="""
predictions = model.predict(test_features)
"""))

final_result = env.step(SubmitAction())
print(f"Final Test Score: {final_result.info['test_score']}")
```

## Action Space

The environment relies on two primary actions:

- **RunCodeAction(code: str)**: Executes a block of Python code in the sandbox.
- **SubmitAction()**: Reads the predictions variable from the sandbox, scores it against the hidden test labels, and terminates the episode.

## Observation Space

At each step, the agent receives a rich Observation object containing:

- **stdout and stderr**: Output from the last executed code.
- **execution_status**: Success, Error, or Timeout.
- **namespace_summary**: A summary of variables currently in the sandbox (shapes, types).
- **validation_score**: The current best score achieved on the validation set.
- **dataset_info**: Metadata about the features, data types, and null counts.