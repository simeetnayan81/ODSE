---
title: Open Data Science Environment (ODSE)
emoji: 🎪
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Open Data Science Sandbox Environment (ODSE)

ODSE is a persistent, sandboxed Reinforcement Learning (RL) environment designed to train and evaluate autonomous AI agents on data science tasks. 

Instead of choosing from a predefined list of discrete actions, agents interact with ODSE by writing and executing raw Python code. The environment evaluates the agent's ability to explore data, handle missing values, train machine learning models, and submit predictions on hidden test sets.

## Key Features

* **Notebook-Style Execution:** Agents execute arbitrary Python code in a persistent namespace. Variables, models, and data frames survive across execution steps.
* **Built-in Datasets:** Includes classification and regression tasks (Breast Cancer, Iris, Wine, House Prices) with configurable difficulty levels (Easy, Medium, Hard) that inject noise and missing values.
* **Safe Sandboxing:** Code is executed with strict wall-clock timeouts, stdout/stderr truncation, and restricted imports (whitelisting `pandas`, `sklearn`, `numpy`, etc.) to prevent environment corruption.
* **Dense Reward Shaping:** Designed for RL. Agents receive immediate heuristic rewards for successful code execution, generating predictions, and improving validation scores, mitigating the sparse-reward problem in code generation.
* **OpenEnv Compatible:** Fully compliant with the OpenEnv standard, enabling easy deployment to Hugging Face Spaces and evaluation via standard inference scripts.

## Quick Start

The simplest way to interact with ODSE via the OpenEnv standard is through the `OdseEnv` client.

```python
import asyncio
from odse import OdseAction, OdseEnv

async def main():
    # Connect to the ODSE environment (local or hosted)
    async with OdseEnv(base_url="https://simeetnayan-odse.hf.space") as env:
        # 1. Reset the environment (starts a new data science task)
        result = await env.reset(difficulty="easy")
        print(result.observation.task_description)

        # 2. Agent explores the data
        code_action = OdseAction(
            action_type="run_code",
            code="print(train_df.head())\nprint(train_df.isnull().sum())"
        )
        result = await env.step(code_action)
        print(result.observation.stdout)

        # 3. Agent trains a model and predicts
        train_code = """
from sklearn.ensemble import RandomForestClassifier
X = train_df.drop(target_column, axis=1)
y = train_df[target_column]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

predictions = model.predict(test_features)
"""
        result = await env.step(OdseAction(action_type="run_code", code=train_code))
        
        # 4. Submit the predictions
        final_result = await env.step(OdseAction(action_type="submit"))
        print(f"Final Test Score: {final_result.info.get('test_score')}")

if __name__ == "__main__":
    asyncio.run(main())
```

That's it! The `OdseEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t odse-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**OdseAction**: Supports two main action types:
- `run_code`: Requires a `code` string containing the Python code to execute.
- `submit`: Terminal action. Evaluates the `predictions` variable on the hidden test set.

### Observation
**OdseObservation**: Contains execution feedback and environment state
- `stdout` / `stderr` (str) - Captured output from the executed code.
- `execution_status` (str) - Result of the last execution (`success`, `error`, `timeout`).
- `namespace_summary` (list) - Active variables and their shapes/types in the sandbox.
- `validation_score` (float) - Current evaluation score on the validation set.
- `dataset_info` (dict) - Metadata about the dataset features and target column.
- `task_description` (str) - Objective description for the agent.

### Reward
ODSE provides both dense and sparse rewards:
- **Dense (Step) Reward**: Heuristic rewards for successful code execution, finding predictions, and improving validation scores (e.g., `+0.05` for success, `-0.05` for errors).
- **Sparse (Final) Reward**: The actual test-set evaluation metric score (e.g., `0.0` to `1.0`) awarded upon submitting.

## Advanced Usage

### Connecting to an Existing Server

If you already have an ODSE environment server running, you can connect directly:

```python
from odse import OdseAction, OdseEnv

# Connect to existing server
odseenv = OdseEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = await odseenv.reset(difficulty="easy")
result = await odseenv.step(OdseAction(action_type="run_code", code="print('Hello ODSE!')"))
```

Note: When connecting to an existing server, `odseenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
import asyncio
from odse import OdseAction, OdseEnv

# Connect with context manager (auto-connects and closes)
async def run_context():
    async with OdseEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(difficulty="medium")
        print(f"Task: {result.observation.task_description}")
        # Multiple steps with low latency
        for code in ["x = 10", "y = 20", "print(x + y)"]:
            result = await env.step(OdseAction(action_type="run_code", code=code))
            print(f"Output: {result.observation.stdout}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps


## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/odse_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
.
├── __init__.py
├── client.py
├── core
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── data_manager.py
│   │   └── datasets.py
│   ├── docker_executor.py
│   ├── Dockerfile.sandbox
│   ├── docs
│   │   └── architecture.md
│   ├── env.py
│   ├── evaluator.py
│   ├── example.py
│   ├── executor.py
│   ├── models.py
│   ├── README.md
│   ├── requirements.txt
│   ├── reward.py
│   ├── sandbox_runner.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server
│   ├── __init__.py
│   ├── app.py
│   ├── odse_environment.py
│   └── requirements.txt
|── uv.lock
```
