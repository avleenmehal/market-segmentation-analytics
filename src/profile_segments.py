import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR  = os.path.join(BASE_DIR, "outputs", "preprocessed")
SEG_DIR   = os.path.join(BASE_DIR, "outputs", "segmentation")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

df_orig = pd.read_pickle(os.path.join(BASE_DIR, "outputs", "clean_data.pkl"))

# reconstruct same row ordering used in preprocess.py so arrays align
idx = np.arange(len(df_orig))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.20, stratify=df_orig["target"], random_state=42)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, stratify=df_orig["target"].iloc[idx_trainval], random_state=42)
idx_full = np.concatenate([idx_train, idx_val, idx_test])

df = df_orig.iloc[idx_full].reset_index(drop=True)

# adults only — same filter as segment.py
adult_mask = df["age"] >= 16
df = df[adult_mask].reset_index(drop=True)

cluster_labels = pd.read_csv(os.path.join(SEG_DIR, "cluster_labels.csv"))["cluster"].values
df["cluster"]    = cluster_labels
df["high_income"] = (df["target"] == 1).astype(int)

N_CLUSTERS    = df["cluster"].nunique()
POP_AVG       = df["high_income"].mean()
CLUSTER_ORDER = sorted(df["cluster"].unique())

print(f"Adults: {len(df):,} | Clusters: {N_CLUSTERS} | Pop income rate: {POP_AVG*100:.2f}%\n")

# load classifier predictions so we can cross-tab cluster vs model flags
X_train_arr = np.load(os.path.join(PREP_DIR, "X_train.npy"))
X_val_arr   = np.load(os.path.join(PREP_DIR, "X_val.npy"))
X_test_arr  = np.load(os.path.join(PREP_DIR, "X_test.npy"))
X_full_arr  = np.vstack([X_train_arr, X_val_arr, X_test_arr])
X_adult     = X_full_arr[adult_mask.values]

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    best_bundle = pickle.load(f)
best_model  = best_bundle["model"]
best_thresh = best_bundle["threshold"]

clf_prob = best_model.predict_proba(X_adult)[:, 1]
clf_pred = (clf_prob >= best_thresh).astype(int)
df["clf_prob"]    = clf_prob
df["clf_flagged"] = clf_pred

COLORS = ["#4C9BE8", "#2ECC71", "#E8724C", "#9B59B6", "#F39C12"]

print("Numerical profile:")
num_stats = {}
for k in CLUSTER_ORDER:
    sub = df[df["cluster"] == k]
    num_stats[k] = {
        "size":              len(sub),
        "pct_of_adults":     round(len(sub) / len(df) * 100, 1),
        "income_rate":       round(sub["high_income"].mean() * 100, 1),
        "median_age":        sub["age"].median(),
        "median_weeks":      sub["weeks worked in year"].median(),
        "pct_fulltime_year": round((sub["weeks worked in year"] == 52).mean() * 100, 1),
        "pct_cap_gains":     round((sub["capital gains"] > 0).mean() * 100, 1),
        "pct_dividends":     round((sub["dividends from stocks"] > 0).mean() * 100, 1),
        "median_employer_sz":sub["num persons worked for employer"].median(),
        "pct_clf_flagged":   round(sub["clf_flagged"].mean() * 100, 1),
        "median_clf_prob":   round(sub["clf_prob"].median(), 3),
    }

num_df = pd.DataFrame(num_stats).T
num_df.index.name = "cluster"
print(num_df.to_string())
print()

def pct_table(col, top_n=5):
    # % of each cluster in each top category
    top_cats = df[col].value_counts().head(top_n).index.tolist()
    rows = {}
    for k in CLUSTER_ORDER:
        sub = df[df["cluster"] == k]
        rows[k] = {cat: round(sub[col].eq(cat).mean() * 100, 1) for cat in top_cats}
    return pd.DataFrame(rows).T

print("Education (top 6):")
edu_pct = pct_table("education", top_n=6)
print(edu_pct.to_string())
print()

print("Occupation (top 5):")
occ_pct = pct_table("major occupation code", top_n=5)
print(occ_pct.to_string())
print()

print("Marital status:")
mar_pct = pct_table("marital stat", top_n=5)
print(mar_pct.to_string())
print()

print("Class of worker:")
cow_pct = pct_table("class of worker", top_n=5)
print(cow_pct.to_string())
print()

print("Sex:")
sex_pct = pct_table("sex", top_n=2)
print(sex_pct.to_string())
print()

# income rate per cluster
fig, ax = plt.subplots(figsize=(8, 4))
rates = [num_stats[k]["income_rate"] for k in CLUSTER_ORDER]
bars  = ax.bar([f"Cluster {k}" for k in CLUSTER_ORDER], rates,
               color=COLORS, edgecolor="white", width=0.6)
ax.axhline(POP_AVG * 100, linestyle="--", color="gray", lw=1.5,
           label=f"Population avg ({POP_AVG*100:.1f}%)")
ax.set_ylabel(">$50k Rate (%)")
ax.set_title("Income Rate by Cluster", fontweight="bold")
ax.legend(fontsize=9)
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "profile_income_rate.png"), dpi=150, bbox_inches="tight")
plt.close()

# age distribution
fig, ax = plt.subplots(figsize=(9, 5))
for k in CLUSTER_ORDER:
    ages = df[df["cluster"] == k]["age"]
    ax.hist(ages, bins=30, alpha=0.55, color=COLORS[k],
            label=f"Cluster {k} (n={len(ages):,})", density=True)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.set_title("Age Distribution by Cluster", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "profile_age_dist.png"), dpi=150, bbox_inches="tight")
plt.close()

# weeks worked
fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(14, 4), sharey=False)
for k, ax in zip(CLUSTER_ORDER, axes):
    weeks = df[df["cluster"] == k]["weeks worked in year"]
    ax.hist(weeks, bins=20, color=COLORS[k], edgecolor="white", alpha=0.85)
    ax.set_title(f"Cluster {k}\n({num_stats[k]['pct_fulltime_year']}% full-year)",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Weeks worked")
    if k == 0:
        ax.set_ylabel("Count")
plt.suptitle("Weeks Worked in Year — by Cluster", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "profile_weeks_worked.png"), dpi=150, bbox_inches="tight")
plt.close()

# education stacked bar
edu_order = [
    "Less than 1st grade", "1st 2nd 3rd or 4th grade", "5th or 6th grade",
    "7th and 8th grade", "9th grade", "10th grade", "11th grade",
    "12th grade no diploma", "High school graduate",
    "Some college but no degree",
    "Associates degree-academic program",
    "Associates degree-occup /vocational",
    "Bachelors degree(BA AB BS)",
    "Masters degree(MA MS MEng MEd MSW MBA)",
    "Prof school degree (MD DDS DVM LLB JD)",
    "Doctorate degree(PhD EdD)",
]
edu_present = [e for e in edu_order if e in df["education"].values]
edu_counts  = (df.groupby(["cluster", "education"])
               .size().unstack(fill_value=0))
edu_counts  = edu_counts[[c for c in edu_present if c in edu_counts.columns]]
edu_pct_plot = edu_counts.div(edu_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 5))
bottom  = np.zeros(N_CLUSTERS)
tab_colors = plt.cm.tab20.colors
for i, col in enumerate(edu_pct_plot.columns):
    ax.bar(range(N_CLUSTERS), edu_pct_plot[col],
           bottom=bottom, color=tab_colors[i % 20],
           label=col, width=0.6)
    bottom += edu_pct_plot[col].values
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([f"Cluster {k}" for k in CLUSTER_ORDER])
ax.set_ylabel("% of Cluster")
ax.set_title("Education Distribution by Cluster", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "profile_education.png"), dpi=150, bbox_inches="tight")
plt.close()

# classifier flag rate per cluster
fig, ax = plt.subplots(figsize=(8, 4))
flag_rates = [num_stats[k]["pct_clf_flagged"] for k in CLUSTER_ORDER]
bars = ax.bar([f"Cluster {k}" for k in CLUSTER_ORDER], flag_rates,
              color=COLORS, edgecolor="white", width=0.6)
overall_flag = round(df["clf_flagged"].mean() * 100, 1)
ax.axhline(overall_flag, linestyle="--", color="gray", lw=1.5,
           label=f"Overall flag rate ({overall_flag}%)")
ax.set_ylabel("% Flagged by Classifier")
ax.set_title("Classifier Flag Rate by Cluster\n(LightGBM, threshold=0.857)",
             fontweight="bold")
ax.legend(fontsize=9)
for bar, val in zip(bars, flag_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "profile_clf_flag_rate.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Plots saved\n")

# cross-tab: cluster × classifier
cross = df.groupby("cluster").agg(
    total=("high_income", "count"),
    actual_pos=("high_income", "sum"),
    clf_flagged=("clf_flagged", "sum"),
    median_prob=("clf_prob", "median"),
).round({"median_prob": 3})
cross["actual_rate_%"]  = (cross["actual_pos"]  / cross["total"] * 100).round(1)
cross["flagged_rate_%"] = (cross["clf_flagged"] / cross["total"] * 100).round(1)
print(cross.to_string())
