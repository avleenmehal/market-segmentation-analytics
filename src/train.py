import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR  = os.path.join(BASE_DIR, "outputs", "preprocessed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
PLOT_DIR  = os.path.join(BASE_DIR, "outputs", "eda")
os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(os.path.join(PREP_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PREP_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(PREP_DIR, "X_test.npy"))

y_train = pd.read_pickle(os.path.join(PREP_DIR, "y_train.pkl")).values
y_val   = pd.read_pickle(os.path.join(PREP_DIR, "y_val.pkl")).values
y_test  = pd.read_pickle(os.path.join(PREP_DIR, "y_test.pkl")).values

with open(os.path.join(PREP_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
SCALE_POS_WEIGHT = n_neg / n_pos
print(f"Train: {len(y_train):,} rows | pos={n_pos:,} ({n_pos/len(y_train)*100:.1f}%) | scale_pos_weight={SCALE_POS_WEIGHT:.2f}\n")

MODELS = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", C=0.1, random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr", early_stopping_rounds=30,
        random_state=42, verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=42, verbose=-1,
    ),
}


def tune_threshold(y_true, y_prob):
    # find threshold that maximises F1 on the given set
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) == 0, 0,
        2 * precision * recall / (precision + recall)
    )
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model":     name,
        "threshold": round(threshold, 3),
        "PR-AUC":    round(average_precision_score(y_true, y_prob), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "F1@0.5":    round(f1_score(y_true, (y_prob >= 0.5).astype(int)), 4),
    }


results = []
trained_models = {}
val_probs = {}

for name, model in MODELS.items():
    print(f"Training {name}...")

    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    prob = model.predict_proba(X_val)[:, 1]
    best_thresh, _ = tune_threshold(y_val, prob)
    metrics = evaluate(name, y_val, prob, best_thresh)
    results.append(metrics)
    trained_models[name] = model
    val_probs[name] = prob

    print(f"  PR-AUC={metrics['PR-AUC']}  ROC-AUC={metrics['ROC-AUC']}  F1={metrics['F1']}  threshold={metrics['threshold']}\n")

results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
print(results_df.to_string(index=False))
print()

best_name  = results_df.iloc[0]["model"]
best_model = trained_models[best_name]
best_thresh = results_df.iloc[0]["threshold"]
print(f"Best model: {best_name} (PR-AUC={results_df.iloc[0]['PR-AUC']})\n")

best_prob_val = val_probs[best_name]
best_pred_val = (best_prob_val >= best_thresh).astype(int)
print(classification_report(y_val, best_pred_val, target_names=["<=50k", ">50k"]))

cm = confusion_matrix(y_val, best_pred_val)
print(f"Confusion matrix (val):  TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}\n")

# PR curves for all models
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#4C9BE8", "#2ECC71", "#E8724C", "#9B59B6"]
for (name, prob), color in zip(val_probs.items(), colors):
    pr, rc, _ = precision_recall_curve(y_val, prob)
    ap = average_precision_score(y_val, prob)
    ax.plot(rc, pr, lw=2, color=color, label=f"{name} (AP={ap:.3f})")
ax.axhline(y_val.mean(), linestyle="--", color="gray", lw=1.2, label=f"Baseline = {y_val.mean():.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Validation Set", fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# feature importance — just LightGBM is enough
for name in ["LightGBM", "XGBoost", "Random Forest"]:
    if name not in trained_models:
        continue
    model = trained_models[name]
    if hasattr(model, "feature_importances_"):
        imp_df = (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4C9BE8", edgecolor="white")
        ax.set_title(f"{name} — Top 20 Feature Importances", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"feature_importance_{name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        break

# save everything
for name, model in trained_models.items():
    fname = name.lower().replace(" ", "_") + ".pkl"
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(model, f)

results_df.to_csv(os.path.join(MODEL_DIR, "val_results.csv"), index=False)

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump({"name": best_name, "model": best_model, "threshold": best_thresh}, f)

print(f"Saved models to {MODEL_DIR}/")
