"""ODSE Environment for Data Cleaning task in Meta OpenEnv."""

import io
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from models import Observation, Action, ImputeAction, DropAction, SubmitAction, StepResult


class ODSEEnvironment:
    """Open Data Science Environment for data cleaning tasks.
    
    Provides a standardized API (reset, state, step) following OpenEnv conventions.
    The agent's goal is to clean data (handle nulls) and improve model accuracy.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed_value = seed
        np.random.seed(seed)
        
        self.original_df: Optional[pd.DataFrame] = None
        self.working_df: Optional[pd.DataFrame] = None
        self.step_count = 0
        self.prev_accuracy = 0.0
        self.initial_accuracy = 0.0
    
    def reset(self) -> Observation:
        """Reset the environment with a dirty Titanic dataset.
        
        Returns:
            Initial observation of the environment.
        """
        # Load the dirty Titanic dataset
        self.original_df = pd.read_csv(io.StringIO(DIRTY_TITANIC_CSV))
        self.working_df = self.original_df.copy()
        self.step_count = 0
        
        # Calculate initial accuracy
        self.prev_accuracy = self._calculate_accuracy()
        self.initial_accuracy = self.prev_accuracy
        
        return self.state()
    
    def state(self) -> Observation:
        """Get the current observation (state) of the environment.
        
        Calculates metadata for all columns and current model accuracy.
        
        Returns:
            Observation with current dataset state and performance metrics.
        """
        # Calculate column metadata
        column_metadata: Dict[str, Dict[str, Any]] = {}
        total_nulls = 0
        
        for col in self.working_df.columns:
            null_count = int(self.working_df[col].isna().sum())
            total_nulls += null_count
            null_percentage = (null_count / len(self.working_df)) * 100
            
            column_metadata[col] = {
                "null_count": null_count,
                "type": str(self.working_df[col].dtype),
                "null_percentage": round(null_percentage, 2),
            }
        
        # Get sample head as JSON
        sample_head = self.working_df.head(5).to_dict(orient="list")
        
        # Calculate current accuracy
        current_accuracy = self._calculate_accuracy()
        
        return Observation(
            column_metadata=column_metadata,
            sample_head=sample_head,
            current_accuracy=current_accuracy,
            step_count=self.step_count,
            nulls_remaining=total_nulls,
        )
    
    def step(self, action: Action) -> StepResult:
        """Execute one step of the environment.
        
        Applies the action, calculates reward, and determines termination.
        
        Args:
            action: The action to execute (ImputeAction, DropAction, or SubmitAction)
            
        Returns:
            StepResult containing observation, reward, done flag, and info dict.
        """
        self.step_count += 1
        
        # Check if action is SubmitAction (terminates the episode)
        if isinstance(action, SubmitAction):
            observation = self.state()
            reward = 0.0  # No additional reward for submit
            return StepResult(
                observation=observation,
                reward=reward,
                done=True,
                info={"reason": "submit_action"},
            )
        
        # Calculate accuracy before action
        old_accuracy = self._calculate_accuracy()
        
        # Apply action to working dataframe
        if isinstance(action, ImputeAction):
            self._apply_impute(action.column, action.strategy)
        elif isinstance(action, DropAction):
            self._apply_drop(action.column)
        
        # Calculate new accuracy
        new_accuracy = self._calculate_accuracy()
        
        # Calculate reward: accuracy improvement * 10 - step penalty
        accuracy_gain = new_accuracy - old_accuracy
        reward = (accuracy_gain * 10.0) - 0.01
        
        # Check termination conditions
        observation = self.state()
        done = observation.nulls_remaining == 0
        
        info = {
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "accuracy_gain": accuracy_gain,
            "reward": reward,
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    
    def _calculate_accuracy(self) -> float:
        """Calculate model accuracy using 5-fold cross-validation with LogisticRegression.
        
        Only trains on complete rows (drops rows with NaN values).
        
        Returns:
            Cross-validation accuracy score (0.0 to 1.0)
        """
        # Drop rows with any NaN values for training
        df_clean = self.working_df.dropna()
        
        # Need at least 4 samples for 5-fold CV
        if len(df_clean) < 5:
            return 0.0
        
        try:
            # Select target and features
            target_col = "Survived"
            feature_cols = [col for col in self.working_df.columns 
                          if col not in [target_col, "PassengerId", "Name", "Ticket", "Cabin"]]
            
            X = df_clean[feature_cols].copy()
            y = df_clean[target_col].copy()
            
            # Encode categorical features
            X_encoded = self._encode_features(X)
            
            # Apply 5-fold cross-validation
            model = LogisticRegression(max_iter=1000, random_state=self.seed_value, solver='lbfgs')
            scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
            
            return float(np.mean(scores))
        except Exception as e:
            # Return 0.0 if accuracy calculation fails
            return 0.0
    
    def _encode_features(self, X: pd.DataFrame) -> np.ndarray:
        """Encode categorical features to numeric values.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Encoded feature array
        """
        X_copy = X.copy()
        
        for col in X_copy.columns:
            if X_copy[col].dtype == "object":
                le = LabelEncoder()
                X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        
        return X_copy.values
    
    def _apply_impute(self, column: str, strategy: str) -> None:
        """Apply imputation strategy to a column.
        
        Args:
            column: Column name
            strategy: Imputation strategy ("mean", "median", "mode")
        """
        if column not in self.working_df.columns:
            return
        
        if strategy == "mean":
            fill_value = self.working_df[column].mean()
            self.working_df[column] = self.working_df[column].fillna(fill_value)
        elif strategy == "median":
            fill_value = self.working_df[column].median()
            self.working_df[column] = self.working_df[column].fillna(fill_value)
        elif strategy == "mode":
            mode_val = self.working_df[column].mode()
            if len(mode_val) > 0:
                self.working_df[column] = self.working_df[column].fillna(mode_val[0])
    
    def _apply_drop(self, column: str) -> None:
        """Drop a column from the working dataframe.
        
        Args:
            column: Column name
        """
        if column in self.working_df.columns:
            self.working_df.drop(columns=[column], inplace=True)


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
