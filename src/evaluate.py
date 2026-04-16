import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR  = os.path.join(BASE_DIR, "outputs", "preprocessed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

X_test = np.load(os.path.join(PREP_DIR, "X_test.npy"))
y_test = pd.read_pickle(os.path.join(PREP_DIR, "y_test.pkl")).values

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    best_bundle = pickle.load(f)

best_name   = best_bundle["name"]
best_model  = best_bundle["model"]
best_thresh = best_bundle["threshold"]  # fixed from val set — not re-tuned on test

print(f"Model: {best_name}  |  threshold (from val): {best_thresh:.3f}")
print(f"Test set: {len(y_test):,} rows | positives: {y_test.sum():,} ({y_test.mean()*100:.2f}%)\n")

test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_thresh).astype(int)

pr_auc  = average_precision_score(y_test, test_prob)
roc_auc = roc_auc_score(y_test, test_prob)
prec    = precision_score(y_test, test_pred, zero_division=0)
rec     = recall_score(y_test, test_pred)
f1      = f1_score(y_test, test_pred)
f1_05   = f1_score(y_test, (test_prob >= 0.5).astype(int))
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
lift    = prec / y_test.mean()

print(f"PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  F1@0.5: {f1_05:.4f}")
print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print(f"Flagged: {tp+fp:,} | Of flagged actually >$50k: {prec*100:.1f}% | Lift: {lift:.1f}x\n")
print(classification_report(y_test, test_pred, target_names=["<=50k", ">50k"]))

# compare against val results
val_results = pd.read_csv(os.path.join(MODEL_DIR, "val_results.csv"))
val_row = val_results[val_results["model"] == best_name].iloc[0]
drift = abs(val_row["PR-AUC"] - pr_auc)
print(f"Val PR-AUC: {val_row['PR-AUC']}  →  Test PR-AUC: {pr_auc:.4f}  (drift={drift:.4f})")
