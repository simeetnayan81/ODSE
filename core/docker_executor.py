"""Docker-based sandbox executor for ODSE.

Executes Python code in isolated Docker containers with strict security constraints:
- Resource limits (CPU, memory)
- Read-only root filesystem (except for specific mounts)
- No network access
- No privilege escalation
- Whitelisted imports enforced
- Timeout enforcement via container timeout
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .executor import ExecutionResult, ALLOWED_MODULES
from .models import ExecutionStatus, VariableInfo


class DockerSandboxExecutor:
    """Executes Python code in isolated Docker containers."""

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_output_chars: int = 10_000,
        docker_image: str = "odse-sandbox:latest",
        memory_limit: str = "512m",
        cpus: float = 1.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.cpus = cpus
        self._namespace: Dict[str, Any] = {}
        self._setup_done: bool = False
        self._work_dir: Optional[Path] = None
        self._namespace_file: Optional[Path] = None
        self._evaluate_fn: Optional[Callable] = None

        # Validate Docker is available
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Docker is not available or not running: {e}. "
                "Ensure Docker is installed and the daemon is running."
            ) from e

    @property
    def namespace(self) -> Dict[str, Any]:
        """Direct (read-only) view of the sandbox namespace."""
        return self._namespace

    def setup_namespace(
        self,
        *,
        train_df: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
        target_column: str,
        evaluate_fn: Callable,
    ) -> None:
        """Initialise the sandbox namespace with pre-loaded variables."""
        self._work_dir = Path(tempfile.mkdtemp(prefix="odse_sandbox_"))
        self._evaluate_fn = evaluate_fn

        self._namespace = {
            "train_df": train_df.copy(),
            "val_features": val_features.copy(),
            "test_features": test_features.copy(),
            "target_column": target_column,
            "pd": pd,
            "np": np,
            "evaluate": None,  # Will be injected as code
            "print": print,
        }

        self._namespace_file = self._work_dir / "namespace.pkl"
        self._save_namespace()
        self._setup_done = True

    def reset(self) -> None:
        """Clear the namespace and clean up temporary files."""
        self._namespace.clear()
        self._setup_done = False
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
        self._namespace_file = None

    def execute(self, code: str) -> ExecutionResult:
        """Execute *code* in a Docker container and return an ``ExecutionResult``."""
        if not self._setup_done:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr="Sandbox not initialised - call setup_namespace() first.",
            )

        start = time.perf_counter()

        try:
            self._save_namespace()

            code_file = self._work_dir / "code.py"
            evaluate_setup_code = self._get_evaluate_setup_code()
            full_code = evaluate_setup_code + "\n\n" + code
            code_file.write_text(full_code, encoding="utf-8")

            cmd = [
                "docker", "run",
                "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpus}",
                "--network=none",
                "--read-only",
                f"--tmpfs=/tmp:size=100m,mode=1777",
                f"--tmpfs=/home:size=100m,mode=1777",
                "-v", f"{self._work_dir}:/sandbox/work:rw",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                self.docker_image,
                "timeout", str(int(self.timeout_seconds) + 5),
                "python", "/app/sandbox_runner.py",
                "/sandbox/work/namespace.pkl",
                "/sandbox/work/code.py",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=int(self.timeout_seconds) + 10,
                text=True,
            )

            elapsed = (time.perf_counter() - start) * 1000

            if result.returncode == 124:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=self._truncate(result.stdout),
                    stderr=f"Code execution exceeded {self.timeout_seconds}s time limit",
                    execution_time_ms=elapsed,
                )

            if self._namespace_file and self._namespace_file.exists():
                self._load_namespace()

            if result.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    stdout=self._truncate(result.stdout),
                    stderr="",
                    execution_time_ms=elapsed,
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stdout="",
                    stderr=self._truncate(result.stderr or result.stdout),
                    execution_time_ms=elapsed,
                )

        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr="Container execution exceeded timeout",
                execution_time_ms=elapsed,
            )

        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=self._truncate(traceback.format_exc()),
                execution_time_ms=elapsed,
            )

    def get_namespace_summary(self) -> List[VariableInfo]:
        """Return a summary of user-visible variables in the namespace."""
        hidden = {"__builtins__", "pd", "np", "evaluate", "target_column", "print"}
        summary: List[VariableInfo] = []
        for name, value in self._namespace.items():
            if name.startswith("_") or name in hidden:
                continue
            summary.append(
                VariableInfo(
                    name=name,
                    type_name=type(value).__name__,
                    shape=getattr(value, "shape", None),
                    preview=self._preview(value),
                )
            )
        return summary

    def get_predictions(self) -> Optional[np.ndarray]:
        """Retrieve ``predictions`` from the namespace (or ``None``)."""
        preds = self._namespace.get("predictions")
        if preds is None:
            return None
        try:
            return np.asarray(preds)
        except Exception:
            return None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.reset()

    # -- Private helpers ----

    def _get_evaluate_setup_code(self) -> str:
        """Generate code to set up the evaluate function in the container."""
        evaluate_fn = self._evaluate_fn
        
        if not (hasattr(evaluate_fn, 'val_labels') and hasattr(evaluate_fn, 'problem_type') and hasattr(evaluate_fn, 'metric')):
            return ""

        problem_type = evaluate_fn.problem_type.value
        metric = evaluate_fn.metric

        # Use string concatenation to avoid nested f-string issues
        # Note: _val_labels will be pre-loaded in the namespace by _save_namespace()
        setup_code = (
            "import numpy as np\n"
            "import sklearn.metrics\n\n"
            "def evaluate(predictions):\n"
            "    preds = np.asarray(predictions)\n"
            "    if len(preds) != len(_val_labels):\n"
            "        return {'error': f'Expected {len(_val_labels)} predictions (val_features length), got {len(preds)}'}\n"
            "    \n"
            "    try:\n"
            + f"        if '{problem_type}' == 'classification':\n"
            "            acc = sklearn.metrics.accuracy_score(_val_labels, preds)\n"
            "            primary = acc\n"
            "            report = {'accuracy': round(acc, 4), 'f1_macro': round(sklearn.metrics.f1_score(_val_labels, preds, average='macro', zero_division=0), 4)}\n"
            "        else:\n"
            "            r2 = sklearn.metrics.r2_score(_val_labels, preds)\n"
            "            primary = r2\n"
            "            report = {'r2': round(r2, 4)}\n"
            "        \n"
            + f"        report['primary_metric'] = '{metric}'\n"
            "        report['primary_score'] = round(primary, 4)\n"
            "        return report\n"
            "    except Exception as e:\n"
            "        return {'error': str(e)}\n"
        )
        return setup_code

    def _save_namespace(self) -> None:
        """Serialize the current namespace to a pickle file."""
        if not self._namespace_file:
            raise RuntimeError("Namespace file not initialized")

        pickleable = {}
        for key, value in self._namespace.items():
            if key in ("print", "evaluate"):
                continue
            try:
                pickle.dumps(value)
                pickleable[key] = value
            except (pickle.PicklingError, TypeError):
                pass

        # Add validation labels to the namespace so the evaluate function can use it
        if self._evaluate_fn and hasattr(self._evaluate_fn, 'val_labels'):
            try:
                pickle.dumps(self._evaluate_fn.val_labels)
                pickleable['_val_labels'] = self._evaluate_fn.val_labels
            except (pickle.PicklingError, TypeError):
                pass

        with open(self._namespace_file, "wb") as f:
            pickle.dump(pickleable, f)

    def _load_namespace(self) -> None:
        """Load the namespace from the pickle file written by the container."""
        if not self._namespace_file or not self._namespace_file.exists():
            return

        try:
            with open(self._namespace_file, "rb") as f:
                loaded = pickle.load(f)
                print_fn = self._namespace.get("print")
                evaluate_fn = self._evaluate_fn
                self._namespace.update(loaded)
                if print_fn:
                    self._namespace["print"] = print_fn
                if evaluate_fn:
                    self._namespace["evaluate"] = evaluate_fn
        except Exception:
            pass

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n... [output truncated]"

    @staticmethod
    def _preview(value: Any, max_len: int = 300) -> str:
        """Generate a short string preview of *value*."""
        try:
            if isinstance(value, pd.DataFrame):
                return f"DataFrame(shape={value.shape}, cols={list(value.columns[:5])})"
            if isinstance(value, pd.Series):
                return f"Series(len={len(value)}, dtype={value.dtype})"
            if isinstance(value, np.ndarray):
                return f"ndarray(shape={value.shape}, dtype={value.dtype})"
            s = repr(value)
            return s[:max_len] if len(s) > max_len else s
        except Exception:
            return "<unprintable>"
