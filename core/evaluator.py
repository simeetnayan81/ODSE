"""Evaluation metrics for ODSE, split by problem type.

Classification : accuracy, f1_macro
Regression     : R^2, RMSE, MAE

Each evaluator computes a *primary* metric (used for reward / scoring)
and a *full report* with all relevant metrics.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

from .models import ProblemType

# ============================================================================
# Public API
# ============================================================================

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: ProblemType,
    metric: str,
) -> float:
    """Compute a single metric value.

    Returns a float where **higher is always better** (RMSE and MAE are
    negated so that improvement means an increase).
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return _classification_metric(y_true, y_pred, metric)
    return _regression_metric(y_true, y_pred, metric)

def compute_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: ProblemType,
) -> Dict[str, Any]:
    """Compute all relevant metrics for *problem_type*."""
    if problem_type == ProblemType.CLASSIFICATION:
        return _classification_report(y_true, y_pred)
    return _regression_report(y_true, y_pred)

# ============================================================================
# Classification
# ============================================================================

def _classification_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true, y_pred = _encode_if_needed(y_true, y_pred)
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # default: accuracy
    return float(accuracy_score(y_true, y_pred))

def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true, y_pred = _encode_if_needed(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
    }

# ============================================================================
# Regression
# ============================================================================

def _regression_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if metric in ("rmse", "neg_rmse"):
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric in ("mae", "neg_mae"):
        return -float(mean_absolute_error(y_true, y_pred))
    # default: r2
    return float(r2_score(y_true, y_pred))

def _regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }

# ============================================================================
# Helpers
# ============================================================================

def _encode_if_needed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-encode if targets are non-numeric (strings / objects)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.dtype.kind in ("U", "S", "O") or y_pred.dtype.kind in ("U", "S", "O"):
        le = LabelEncoder()
        combined = np.concatenate([y_true.ravel(), y_pred.ravel()])
        le.fit(combined.astype(str))
        y_true = le.transform(y_true.astype(str))
        y_pred = le.transform(y_pred.astype(str))
    return y_true, y_pred