"""
Trains a small logistic regression confidence calibrator on collected batch_eval 
results, using ncc_score and ambiguity_ratio as features to predict success/failure.
This is a bonus/comparison model -- it does not replace the existing NCC matcher 
or the fixed ambiguity-ratio threshold.

Usage:
    python train_calibrator.py
"""
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

FAIL_THRESHOLD_PX = 50

# Load all collected CSVs
csv_files = glob.glob("training_data/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSVs found in training_data/ -- run Step 2 first.")

print(f"Loading {len(csv_files)} files: {csv_files}")
dfs = [pd.read_csv(f) for f in csv_files]
data = pd.concat(dfs, ignore_index=True)
print(f"Total examples: {len(data)}")

# Label: success if pixel_error under threshold, else failure
data["label"] = (data["pixel_error"] <= FAIL_THRESHOLD_PX).astype(int)
print(f"Success examples: {data['label'].sum()}, Failure examples: {(data['label']==0).sum()}")

# Features: ncc_score and ambiguity_ratio
X = data[["ncc_score", "ambiguity_ratio"]].values
y = data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Trained Calibrator Results ---")
print(f"Test accuracy: {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["FAIL", "SUCCESS"]))

# Compare against the fixed threshold rule (ratio > 0.995 -> LOW/fail)
fixed_rule_pred = (data.loc[X_test.shape[0]*0:, "ambiguity_ratio"].values <= 0.995).astype(int)
# (using full data column aligned isn't exact since split shuffled -- simpler: recompute on test set directly)
test_ratios = X_test[:, 1]
fixed_rule_pred = (test_ratios <= 0.995).astype(int)
fixed_acc = accuracy_score(y_test, fixed_rule_pred)
print(f"\nFixed threshold (0.995) rule accuracy on same test set: {fixed_acc:.3f}")
print(f"Learned model accuracy: {acc:.3f}")

if acc > fixed_acc:
    print("\n-> Learned model outperforms the fixed threshold.")
elif acc < fixed_acc:
    print("\n-> Fixed threshold performs better -- learned model does not improve on it.")
else:
    print("\n-> Learned model and fixed threshold perform equivalently.")

# Save the trained model
joblib.dump(model, "calibrator_model.pkl")
print("\nSaved trained model to calibrator_model.pkl")
print(f"Learned coefficients: ncc_score={model.coef_[0][0]:.3f}, ambiguity_ratio={model.coef_[0][1]:.3f}")
print(f"Intercept: {model.intercept_[0]:.3f}")