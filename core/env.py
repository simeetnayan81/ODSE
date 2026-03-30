"""ODS Environment - the main entry-point for RL agents.

Follows the strict API: ``reset()``, ``state()``, and ``step(action)``.
All domain-specific logic is delegated to the active *task* object,
making the environment fully task-agnostic and extensible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .models import (                                                                                                                                                                                                                                                                           
    Action,
    Difficulty,
    Observation,
    StepResult,
    TaskType,
)
from .tasks.base_task import BaseTask
from .tasks.registry import create_task

class ODSEnvironment:
    """Open Data Science Environment
    
    Provides a standardized RL API around data-science tasks
    (cleaning, feature engineering, and future extensions).

    Parameters
    ----------
    task_type
        Which task to run (e.g.  ``"data_cleaning"`` or TaskType.DATA_CLEANING).
    difficulty
        Difficulty level of the task (e.g. ``"easy"``, ``"medium"`, or ``"hard"``).
    seed
        RNG seed for reproducibility.
    
    Example
    -------
    >>> env = ODSEnvironment(task_type="data_cleaning", difficulty="medium", seed=42)
    >>> obs = env.reset()
    >>> result = env.step(ImputeAction(column="Age", strategy="mean"))
    >>> print(result.reward)
    """


def grade_performance(final_df: pd.DataFrame, env: Optional[ODSEEnvironment] = None) -> float:
    """Grade the final performance of data cleaning.
    
    Scoring:
        - 0.5 points: All nulls are removed (no missing values)
        - 0.5 points: Final accuracy is > 0.78
    
    Args:
        final_df: Final cleaned dataframe
        env: Optional environment reference for context
        
    Returns:
        Float score from 0.0 to 1.0
    """
    score = 0.0
    
    # Check 1: Zero nulls (0.5 points)
    if final_df.isna().sum().sum() == 0:
        score += 0.5
    
    # Check 2: Accuracy > 0.78 (0.5 points)
    # We need to evaluate the final model accuracy
    df_clean = final_df.dropna()
    
    if len(df_clean) >= 5:
        try:
            target_col = "Survived"
            feature_cols = [col for col in final_df.columns 
                          if col not in [target_col, "PassengerId", "Name", "Ticket", "Cabin"]]
            
            if target_col in df_clean.columns and len(feature_cols) > 0:
                X = df_clean[feature_cols].copy()
                y = df_clean[target_col].copy()
                
                # Encode categorical features
                X_encoded = np.full((len(X), len(feature_cols)), fill_value=0.0)
                
                for idx, col in enumerate(feature_cols):
                    if X[col].dtype == "object":
                        le = LabelEncoder()
                        X_encoded[:, idx] = le.fit_transform(X[col].astype(str))
                    else:
                        X_encoded[:, idx] = X[col].values
                
                model = LogisticRegression(max_iter=1000, solver='lbfgs')
                scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
                final_accuracy = float(np.mean(scores))
                
                if final_accuracy > 0.78:
                    score += 0.5
        except Exception:
            pass
    
    return min(score, 1.0)  # Cap at 1.0
