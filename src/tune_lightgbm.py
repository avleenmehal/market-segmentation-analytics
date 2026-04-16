import os
import pickle
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR  = os.path.join(BASE_DIR, "outputs", "preprocessed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

X_train = np.load(os.path.join(PREP_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PREP_DIR, "X_val.npy"))
y_train = pd.read_pickle(os.path.join(PREP_DIR, "y_train.pkl")).values
y_val   = pd.read_pickle(os.path.join(PREP_DIR, "y_val.pkl")).values

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
SCALE_POS_WEIGHT = n_neg / n_pos

# round 1 — informed defaults based on dataset characteristics
baseline_params = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, verbose=-1,
)
lgbm_base = LGBMClassifier(**baseline_params)
lgbm_base.fit(X_train, y_train)
base_val_prauc = average_precision_score(y_val, lgbm_base.predict_proba(X_val)[:, 1])
print(f"Round 1 val PR-AUC: {base_val_prauc:.4f}\n")

# search space centred around round 1 values — checking whether we were already close
param_dist = {
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "num_leaves":       [31, 63, 95, 127],
    "max_depth":        [5, 6, 7, 8, -1],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "n_estimators":     [300, 400, 500],
    "min_child_samples":[10, 20, 30],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=LGBMClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, verbose=-1),
    param_distributions=param_dist,
    n_iter=25,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True,
)

print("Running RandomizedSearchCV (25 combinations, 3-fold CV)...")
search.fit(X_train, y_train)

results_df = (
    pd.DataFrame(search.cv_results_)
    .sort_values("mean_test_score", ascending=False)
    [["mean_test_score", "std_test_score",
      "param_learning_rate", "param_num_leaves", "param_max_depth",
      "param_subsample", "param_colsample_bytree",
      "param_n_estimators", "param_min_child_samples"]]
    .rename(columns={"mean_test_score": "cv_prauc", "std_test_score": "cv_std"})
    .round({"cv_prauc": 4, "cv_std": 4})
    .reset_index(drop=True)
)

print("\nTop 10 combinations:")
print(results_df.head(10).to_string(index=False))

best_params    = search.best_params_
tuned_model    = search.best_estimator_
tuned_prauc    = average_precision_score(y_val, tuned_model.predict_proba(X_val)[:, 1])
improvement    = tuned_prauc - base_val_prauc

print(f"\nRound 1 val PR-AUC : {base_val_prauc:.4f}")
print(f"Tuned val PR-AUC   : {tuned_prauc:.4f}")
print(f"Improvement        : {improvement:+.4f}")

# only retrain if the gain is meaningful — marginal improvement isn't worth the complexity
IMPROVEMENT_THRESHOLD = 0.003

if improvement > IMPROVEMENT_THRESHOLD:
    print(f"\nImprovement exceeds threshold ({IMPROVEMENT_THRESHOLD}) — retraining with tuned params")
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

    probs_val = tuned_model.predict_proba(X_val)[:, 1]
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, probs_val)
    f1_arr = np.where(
        (precision_arr + recall_arr) == 0, 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr)
    )
    best_thresh = thresholds[np.argmax(f1_arr[:-1])]
    y_pred = (probs_val >= best_thresh).astype(int)

    print(f"Threshold: {best_thresh:.3f} | F1: {f1_score(y_val, y_pred):.4f} | "
          f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f} | "
          f"Recall: {recall_score(y_val, y_pred):.4f}")

    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({"name": "LightGBM (tuned)", "model": tuned_model, "threshold": best_thresh}, f)
    with open(os.path.join(MODEL_DIR, "lightgbm.pkl"), "wb") as f:
        pickle.dump(tuned_model, f)

    print("Updated best_model.pkl and lightgbm.pkl")
else:
    print(f"\nImprovement ({improvement:+.4f}) below threshold — round 1 defaults validated, no retraining")

results_df.to_csv(os.path.join(MODEL_DIR, "lgbm_tuning_results.csv"), index=False)
print(f"Tuning results saved to {MODEL_DIR}/lgbm_tuning_results.csv")
