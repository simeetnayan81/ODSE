"""Example usage of the ODSE Sandbox Environment.

Run with:  python -m core.example   (from the ODSE root directory)

Demonstrates:
  1. Classification (Breast Cancer) - explore, train, evaluate, submit
  2. Regression (House Price) - full pipeline in fewer steps
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ODSE package root is on sys.path when running as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.env import ODSEnvironment
from core.models import RunCodeAction, SubmitAction

# ============================================================================
# Helper
# ============================================================================

def _print_step(label: str, result) -> None:
    obs = result.observation
    print(f"  [{label}]")
    print(f"    status : {obs.execution_status}")
    print(f"    reward : {result.reward:+.3f}")
    if obs.stdout:
        # Show first 300 chars of stdout
        preview = obs.stdout[:300].rstrip()
        print(f"    stdout : {preview}")
    if obs.stderr:
        print(f"    stderr : {obs.stderr[:200].rstrip()}")
    if obs.validation_score is not None:
        print(f"    val_score : {obs.validation_score:.4f}")
    print()

# ============================================================================
# Classification example
# ============================================================================

def run_classification_example() -> None:
    """Run a classification episode on the Breast Cancer dataset."""
    print("=" * 64)
    print("  CLASSIFICATION EXAMPLE - Breast Cancer (easy)")
    print("=" * 64)

    env = ODSEnvironment(dataset="breast_cancer", difficulty="easy", seed=42)
    obs = env.reset()

    di = obs.dataset_info
    print(f"\n  Dataset       : breast_cancer (easy)")
    print(f"  Train shape   : {di.train_shape}")
    print(f"  Val shape     : {di.val_shape}")
    print(f"  Test shape    : {di.test_shape}")
    print(f"  Target        : {di.target_column} ({di.problem_type})")
    print(f"  Metric        : {di.metric}")
    print(f"  Max steps     : {obs.max_steps}")
    print(f"\n  Task:\n  {obs.task_description}\n")

    # -- Step 1: Explore -----------------------------------------------------
    r = env.step(RunCodeAction(code="""\
print("== Training Data ==")
print(f"Shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"\\nNull counts:\\n{train_df.isnull().sum()}")
print(f"\\nTarget distribution:\\n{train_df[target_column].value_counts()}")
"""))
    _print_step("Step 1 - Explore data", r)

    # -- Step 2: Train a model -----------------------------------------------
    r = env.step(RunCodeAction(code="""\
from sklearn.linear_model import LogisticRegression

# Prepare training data (all numeric, no encoding needed)
X_train = train_df.drop(target_column, axis=1).copy()
y_train = train_df[target_column]

# Fill any missing values
X_train = X_train.fillna(X_train.median(numeric_only=True))

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"Model trained: {X_train.shape[0]} samples, {X_train.shape[1]} features")
"""))
    _print_step("Step 2 - Train model", r)

    # -- Step 3: Evaluate on validation --------------------------------------
    r = env.step(RunCodeAction(code="""\
# Prepare validation features (same transforms)
X_val = val_features.copy()
X_val = X_val.fillna(X_val.median(numeric_only=True))

val_preds = model.predict(X_val)
score = evaluate(val_preds)
print(f"Validation report: {score}")
"""))
    _print_step("Step 3 - Evaluate on validation", r)

    # -- Step 4: Predict on test and submit ----------------------------------
    r = env.step(RunCodeAction(code="""\
# Prepare test features (same transforms)
X_test = test_features.copy()
X_test = X_test.fillna(X_test.median(numeric_only=True))

predictions = model.predict(X_test)
print(f"Test predictions shape: {predictions.shape}")
"""))
    _print_step("Step 4 - Predict on test set", r)

    r = env.step(SubmitAction())
    print(f"  [Submit]")
    print(f"    final reward : {r.reward:.4f}")
    print(f"    test_score   : {r.info.get('test_score')}")
    print(f"    test_report  : {r.info.get('test_report')}")
    print(f"    steps_taken  : {r.info.get('steps_taken')}")
    print(f"    done         : {r.done}")
    print()


# ============================================================================
# Regression example
# ============================================================================

def run_regression_example() -> None:
    """Run a regression episode on the House Price dataset."""
    print("=" * 64)
    print("  REGRESSION EXAMPLE - House Price (easy)")
    print("=" * 64)

    env = ODSEnvironment(dataset="house_price", difficulty="easy", seed=42)
    obs = env.reset()

    di = obs.dataset_info
    print(f"\n  Dataset       : house_price (easy)")
    print(f"  Train shape   : {di.train_shape}")
    print(f"  Target        : {di.target_column} ({di.problem_type})")
    print(f"  Metric        : {di.metric}")
    print(f"  Target stats  : {di.target_stats}")
    print()

    # -- Step 1: Full pipeline in one cell -----------------------------------
    r = env.step(RunCodeAction(code="""\
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Prepare
X = train_df.drop(target_column, axis=1).copy()
y = train_df[target_column]

# Encode categoricals (e.g. neighborhood)
enc = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    enc[col] = le

X = X.fillna(X.median(numeric_only=True))

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Validate
Xv = val_features.copy()
for col in Xv.select_dtypes(include=['object', 'category']).columns:
    if col in enc:
        Xv[col] = enc[col].transform(Xv[col].astype(str))
    else:
        le = LabelEncoder()
        Xv[col] = le.fit_transform(Xv[col].astype(str))
Xv = Xv.fillna(Xv.median(numeric_only=True))
val_preds = model.predict(Xv)
print(f"Validation: {evaluate(val_preds)}")

# Test predictions
Xt = test_features.copy()
for col in Xt.select_dtypes(include=['object', 'category']).columns:
    if col in enc:
        Xt[col] = enc[col].transform(Xt[col].astype(str))
    else:
        le = LabelEncoder()
        Xt[col] = le.fit_transform(Xt[col].astype(str))
Xt = Xt.fillna(Xt.median(numeric_only=True))
predictions = model.predict(Xt)
print(f"Test predictions: {predictions.shape}")
"""))
    _print_step("Step 1 - Full pipeline", r)

    # -- Submit --------------------------------------------------------------
    r = env.step(SubmitAction())
    print(f"  [Submit]")
    print(f"    final reward : {r.reward:.4f}")
    print(f"    test_report  : {r.info.get('test_report')}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    run_classification_example()
    print("\n" + "-" * 64 + "\n")
    run_regression_example()