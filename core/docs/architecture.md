# ODSE Architecture

The ODSE codebase is structured around a central environment loop, supported by isolated modules for code execution, data management, evaluation, and strongly-typed data models.



## 1. Core Environment (`env.py`)
The `ODSEnvironment` class is the main entry point and orchestrator. It adheres to a standard RL interface (`reset`, `step`, `state`).
* **State Management:** It holds the current data split, the instance of the executor, and tracks episode progression (step counts, best validation scores).
* **Action Routing:** It inspects the incoming `Action` (via Pydantic discriminators) and routes it to either `_handle_run_code()` or `_handle_submit()`.
* **Observation Construction:** It builds the complex `Observation` object, querying the executor for namespace summaries and combining it with dataset metadata.

## 2. Sandbox Execution (`executor.py`)
The `SandboxExecutor` is responsible for safely running the agent's Python code.
* **Persistent Namespace:** It maintains a standard Python dictionary (`self._namespace`) passed to `exec()`. This mimics a Jupyter Notebook kernel.
* **Security Constraints:** * Replaces `__builtins__` with a safe dictionary (blocking `eval`, `open`, `input`, etc.).
  * Overrides `__import__` to strictly enforce an `ALLOWED_MODULES` whitelist (e.g., `sklearn`, `pandas`).
* **Resource Limits:** Uses `signal.SIGALRM` to enforce strict wall-clock time limits on execution, preventing infinite loops. Captures and truncates `stdout` and `stderr` using `io.StringIO`.

## 3. Data Pipeline (`data_manager.py` & `datasets.py`)
Responsible for loading and preparing the scenarios the agent will face.
* **Datasets:** Fetches base datasets (Iris, Breast Cancer, Synthetic, House Prices).
* **Difficulty Scaling:** Based on the requested difficulty (`EASY`, `MEDIUM`, `HARD`), it injects realistic data science hurdles: missing values (`NaN` injection) and categorical noise columns.
* **Data Split:** `DataSplit` rigorously partitions data into Train (features + labels), Validation (features only; labels hidden in the environment), and Test (features only; labels hidden for final scoring).

## 4. Evaluation and Rewards (`evaluator.py` & `reward.py`)
* **Evaluator:** Calculates primary and secondary metrics based on the `ProblemType`. (e.g., Accuracy and F1-Macro for Classification; R², RMSE, MAE for Regression). Includes automated label encoding if an agent outputs categorical strings instead of numeric codes.
* **Reward Shaping:** Computes both dense and sparse rewards:
  * **Dense:** Calculated per `RunCodeAction`. Includes step penalties, execution success bonuses, and proportional rewards for improving the hidden validation score.
  * **Sparse:** Calculated on `SubmitAction`. The final test set metric.

## 5. Data Contracts (`models.py`)
Uses `Pydantic` to define strict schemas for all inputs and outputs.
* **Actions:** `RunCodeAction` and `SubmitAction`.
* **Observations:** Structures the feedback loop (`Observation`, `DatasetInfo`, `ColumnSchema`, `VariableInfo`).
* **Enums:** `ProblemType`, `Difficulty`, `ExecutionStatus`.

This decoupling ensures that the environment is easily extensible. Adding a new dataset or a new evaluation metric requires touching only one specific file without disrupting the core RL loop.