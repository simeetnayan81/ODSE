"""Sandboxed Python code executor for ODSE.

Executes agent-written code in a restricted namespace with:
- Whitelisted imports (pandas, numpy, sklearn, scipy, etc.)
- Per-execution time limits
- Captured stdout / stderr
- Persistent namespace across calls (notebook-kernel semantics)
"""

from __future__ import annotations

import builtins
import io
import signal
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .models import ExecutionStatus, VariableInfo


# ============================================================================
# Security: allowed imports and blocked builtins
# ============================================================================

ALLOWED_MODULES: set[str] = {
    # Core data-science stack
    "numpy", "pandas", "sklearn", "scipy",
    "math", "statistics",
    # Standard-library utilities
    "collections", "itertools", "functools",
    "re", "json", "copy", "typing", "operator",
    "datetime", "time", "warnings",
    # sklearn sub-packages (non-exhaustive, top-level covers them)
    "sklearn.linear_model", "sklearn.ensemble", "sklearn.tree",
    "sklearn.svm", "sklearn.neighbors", "sklearn.naive_bayes",
    "sklearn.preprocessing", "sklearn.model_selection",
    "sklearn.metrics", "sklearn.pipeline", "sklearn.impute",
    "sklearn.decomposition", "sklearn.cluster",
    "sklearn.feature_selection", "sklearn.feature_extraction",
    # scipy sub-packages
    "scipy.stats", "scipy.sparse", "scipy.optimize",
    # Optional extras
    "xgboost", "lightgbm", "catboost",
}

BLOCKED_BUILTINS: set[str] = {
    "exec", "eval", "compile",  # We provide safe alternatives
    "__import__",               # Replaced by _safe_import
    "open", "input",            # No file/terminal I/O
    "breakpoint", "exit", "quit",
}

# ============================================================================
# Execution result (internal data class)
# ============================================================================

class ExecutionResult:
    """Immutable result of a single code execution."""

    __slots__ = ("status", "stdout", "stderr", "execution_time_ms")

    def __init__(
        self,
        status: ExecutionStatus,
        stdout: str = "",
        stderr: str = "",
        execution_time_ms: float = 0.0,
    ) -> None:
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time_ms = execution_time_ms

# ============================================================================
# Sandbox executor
# ============================================================================

class _SandboxTimeout(Exception):
    """Raised when code execution exceeds the time limit."""


class SandboxExecutor:
    """Executes Python code in a sandboxed, persistent namespace.

    Simulates a Jupyter-notebook-style kernel: variables created in one
    ``execute()`` call are visible in subsequent calls.

    Parameters
    ----------
    timeout_seconds : float
        Maximum wall-clock time per ``execute()`` call.
    max_output_chars : int
        Stdout/stderr truncation threshold.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_output_chars: int = 10_000,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self._namespace: Dict[str, Any] = {}
        self._setup_done: bool = False

    # -- Properties ----------------------------------------------------------

    @property
    def namespace(self) -> Dict[str, Any]:
        """Direct (read-only) view of the sandbox namespace."""
        return self._namespace

    # -- Lifecycle -----------------------------------------------------------

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
        self._namespace = {
            # Data
            "train_df": train_df.copy(),
            "val_features": val_features.copy(),
            "test_features": test_features.copy(),
            "target_column": target_column,
            # Libraries
            "pd": pd,
            "np": np,
            # Evaluation helper
            "evaluate": evaluate_fn,
            # print is captured via redirect_stdout
            "print": print,
        }
        self._namespace["__builtins__"] = self._make_safe_builtins()
        self._setup_done = True

    def reset(self) -> None:
        """Clear the namespace entirely."""
        self._namespace.clear()
        self._setup_done = False

    # -- Code execution ------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        """Execute *code* in the sandbox and return an ``ExecutionResult``."""
        if not self._setup_done:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr="Sandbox not initialised - call setup_namespace() first.",
            )

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.perf_counter()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                self._exec_with_timeout(code)

            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=self._truncate(stdout_buf.getvalue()),
                stderr=self._truncate(stderr_buf.getvalue()),
                execution_time_ms=elapsed,
            )

        except _SandboxTimeout as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout=self._truncate(stdout_buf.getvalue()),
                stderr=str(exc),
                execution_time_ms=elapsed,
            )

        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout=self._truncate(stdout_buf.getvalue()),
                stderr=self._truncate(traceback.format_exc()),
                execution_time_ms=elapsed,
            )

    # -- Introspection -------------------------------------------------------

    def get_namespace_summary(self) -> List[VariableInfo]:
        """Return a summary of user-visible variables in the namespace."""
        # Variables injected by the environment that agents shouldn't inspect
        hidden = {
            "__builtins__", "pd", "np", "evaluate",
            "target_column", "print",
        }
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

    # -- Private helpers -----------------------------------------------------

    def _exec_with_timeout(self, code: str) -> None:
        """Compile and exec *code* with an optional SIGALRM timeout."""
        compiled = compile(code, "<sandbox>", "exec")

        old_handler = None
        if hasattr(signal, "SIGALRM"):
            def _alarm(signum, frame):  # noqa: ARG001
                raise _SandboxTimeout(
                    f"Code execution exceeded {self.timeout_seconds}s time limit"
                )
            old_handler = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(int(self.timeout_seconds))

        try:
            exec(compiled, self._namespace)  # noqa: S102
        finally:
            if hasattr(signal, "SIGALRM") and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _make_safe_builtins(self) -> Dict[str, Any]:
        """Build a restricted ``__builtins__`` dict."""
        safe: Dict[str, Any] = {}
        for name in dir(builtins):
            if name not in BLOCKED_BUILTINS:
                safe[name] = getattr(builtins, name)
        # Provide a guarded import
        safe["__import__"] = self._safe_import
        return safe

    def _safe_import(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """``__import__`` replacement that only allows whitelisted modules."""
        top_level = name.split(".")[0]
        if name in ALLOWED_MODULES or top_level in ALLOWED_MODULES:
            return __import__(name, *args, **kwargs)
        raise ImportError(
            f"Module '{name}' is not allowed in the sandbox. "
            f"Allowed top-level modules: "
            f"{', '.join(sorted({m.split('.')[0] for m in ALLOWED_MODULES}))}"
        )

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