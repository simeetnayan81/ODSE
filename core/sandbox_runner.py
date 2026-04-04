#!/usr/bin/env python3
"""Runner script for sandboxed code execution inside Docker containers.

This script:
1. Loads the pickled namespace from disk
2. Loads the evaluate function if available from a separate file
3. Executes user-provided code with security constraints
4. Captures stdout/stderr
5. Saves the updated namespace back to disk
6. Returns appropriate exit codes
"""

import io
import pickle
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Security: allowed imports and blocked builtins
ALLOWED_MODULES = {
    # Core data-science stack
    "numpy", "pandas", "sklearn", "scipy",
    "math", "statistics",
    # Standard-library utilities
    "collections", "itertools", "functools",
    "re", "json", "copy", "typing", "operator",
    "datetime", "time", "warnings", "pickle",
    # sklearn sub-packages
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

BLOCKED_BUILTINS = {
    "exec", "eval", "compile",
    "__import__",
    "open", "input",
    "breakpoint", "exit", "quit",
}


def safe_import(name, *args, **kwargs):
    """Import guard that only allows whitelisted modules."""
    top_level = name.split(".")[0]
    if name in ALLOWED_MODULES or top_level in ALLOWED_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(
        f"Module '{name}' is not allowed in the sandbox. "
        f"Allowed top-level modules: "
        f"{', '.join(sorted({m.split('.')[0] for m in ALLOWED_MODULES}))}"
    )


def make_safe_builtins():
    """Build a restricted __builtins__ dict."""
    import builtins
    safe = {}
    for name in dir(builtins):
        if name not in BLOCKED_BUILTINS:
            safe[name] = getattr(builtins, name)
    safe["__import__"] = safe_import
    return safe


def main():
    """Main execution routine."""
    if len(sys.argv) != 3:
        print("Usage: runner.py <namespace_file> <code_file>", file=sys.stderr)
        return 1

    namespace_file = sys.argv[1]
    code_file = sys.argv[2]

    # Load the namespace
    try:
        with open(namespace_file, "rb") as f:
            namespace = pickle.load(f)
    except Exception as e:
        print(f"Failed to load namespace: {e}", file=sys.stderr)
        return 1

    # Try to load evaluate function from a separate file
    evaluate_file = Path(namespace_file).parent / "evaluate.pkl"
    if evaluate_file.exists():
        try:
            with open(evaluate_file, "rb") as f:
                namespace["evaluate"] = pickle.load(f)
        except Exception:
            pass

    # Set up safe builtins
    namespace["__builtins__"] = make_safe_builtins()

    # Read the code
    try:
        with open(code_file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Failed to read code file: {e}", file=sys.stderr)
        return 1

    # Compile and execute the code
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        compiled = compile(code, "<sandbox>", "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, namespace)  # noqa: S102

        # Print captured output
        stdout_content = stdout_buf.getvalue()
        stderr_content = stderr_buf.getvalue()
        if stdout_content:
            print(stdout_content, end="")
        if stderr_content:
            print(stderr_content, file=sys.stderr, end="")

    except Exception as e:
        # Print error traceback
        stdout_content = stdout_buf.getvalue()
        stderr_content = stderr_buf.getvalue()
        if stdout_content:
            print(stdout_content, end="")
        if stderr_content:
            print(stderr_content, file=sys.stderr, end="")
        print(traceback.format_exc(), file=sys.stderr)
        # Save namespace anyway (for debugging)
        try:
            pickleable = {}
            for key, value in namespace.items():
                if key == "__builtins__":
                    continue
                try:
                    pickle.dumps(value)
                    pickleable[key] = value
                except (pickle.PicklingError, TypeError):
                    pass
            with open(namespace_file, "wb") as f:
                pickle.dump(pickleable, f)
        except Exception:
            pass
        return 1

    # Save the updated namespace
    try:
        pickleable = {}
        for key, value in namespace.items():
            if key == "__builtins__":
                continue
            try:
                pickle.dumps(value)
                pickleable[key] = value
            except (pickle.PicklingError, TypeError):
                pass
        with open(namespace_file, "wb") as f:
            pickle.dump(pickleable, f)
    except Exception as e:
        print(f"Failed to save namespace: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
