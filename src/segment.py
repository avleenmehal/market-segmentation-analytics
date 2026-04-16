import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR = os.path.join(BASE_DIR, "outputs", "preprocessed")
SEG_DIR  = os.path.join(BASE_DIR, "outputs", "segmentation")
os.makedirs(SEG_DIR, exist_ok=True)

X_train = np.load(os.path.join(PREP_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(PREP_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(PREP_DIR, "X_test.npy"))
X_full  = np.vstack([X_train, X_val, X_test])

y_train = pd.read_pickle(os.path.join(PREP_DIR, "y_train.pkl")).values
y_val   = pd.read_pickle(os.path.join(PREP_DIR, "y_val.pkl")).values
y_test  = pd.read_pickle(os.path.join(PREP_DIR, "y_test.pkl")).values
y_full  = np.concatenate([y_train, y_val, y_test])

df_orig = pd.read_pickle(os.path.join(BASE_DIR, "outputs", "clean_data.pkl"))

# reconstruct the same row ordering used in preprocess.py so arrays align
idx = np.arange(len(df_orig))
idx_trainval, idx_test = train_test_split(idx, test_size=0.20, stratify=df_orig["target"], random_state=42)
idx_train, idx_val     = train_test_split(idx_trainval, test_size=0.25, stratify=df_orig["target"].iloc[idx_trainval], random_state=42)
idx_full = np.concatenate([idx_train, idx_val, idx_test])
df_profile = df_orig.iloc[idx_full].reset_index(drop=True)

# filter to adults — children (age<16) are 25% of dataset with 0% income rate
# and 'Not in universe' for all employment features; they'd dominate clusters trivially
adult_mask = df_profile["age"] >= 16
X_full     = X_full[adult_mask.values]
y_full     = y_full[adult_mask.values]
df_profile = df_profile[adult_mask].reset_index(drop=True)

print(f"Adults only: {X_full.shape} | positive rate: {y_full.mean()*100:.2f}%\n")

# PCA — K-Means distance degrades in high dimensions; compress to 90% variance
pca_full = PCA(random_state=42)
pca_full.fit(X_full)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, 0.90)) + 1
print(f"PCA: {n_components} components → {cumvar[n_components-1]*100:.1f}% variance\n")

pca   = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_full)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cumvar)+1), cumvar * 100, lw=2, color="#4C9BE8")
ax.axhline(90, linestyle="--", color="gray", lw=1.2, label="90% threshold")
ax.axvline(n_components, linestyle="--", color="#E8724C", lw=1.2, label=f"{n_components} components")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA — Cumulative Explained Variance", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "pca_variance.png"), dpi=150, bbox_inches="tight")
plt.close()

# sweep k=3..7 — cap at 7, more segments than that are rarely actionable for marketing
K_RANGE     = range(3, 8)
inertias    = []
silhouettes = []

print("K-Means sweep:")
for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca, labels, sample_size=10000, random_state=42)
    silhouettes.append(sil)
    print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_RANGE), inertias, "o-", lw=2, color="#4C9BE8")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Curve", fontweight="bold")
ax2.plot(list(K_RANGE), silhouettes, "o-", lw=2, color="#2ECC71")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontweight="bold")
plt.suptitle("K-Means — Choosing k", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "kmeans_k_selection.png"), dpi=150, bbox_inches="tight")
plt.close()

# silhouette peaks at k=7 (0.437), but k=5 (0.411) is only 0.026 lower
# 5 segments is the practical max for a retail team — 7 creates naming/overlap issues
auto_k = list(K_RANGE)[np.argmax(silhouettes)]
best_k = 5
print(f"\nAuto k={auto_k} (silhouette peak), using k={best_k} for marketing interpretability\n")

km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
cluster_labels = km_final.fit_predict(X_pca)

with open(os.path.join(SEG_DIR, "kmeans_model.pkl"), "wb") as f:
    pickle.dump({"model": km_final, "pca": pca, "k": best_k}, f)

df_profile["cluster"]     = cluster_labels
df_profile["high_income"] = y_full

# drop migration cols that were removed in preprocessing
DROP_COLS = ["migration code-change in msa", "migration code-change in reg",
             "migration code-move within reg", "migration prev res in sunbelt",
             "weight", "year"]
df_profile = df_profile.drop(columns=[c for c in DROP_COLS if c in df_profile.columns])

size_df = (
    df_profile.groupby("cluster")
    .agg(size=("cluster", "count"),
         pct_of_total=("cluster", lambda x: len(x) / len(df_profile) * 100),
         high_income_rate=("high_income", "mean"))
    .round({"pct_of_total": 1, "high_income_rate": 4})
    .reset_index()
)
size_df["high_income_rate_pct"] = (size_df["high_income_rate"] * 100).round(1)
print(size_df[["cluster", "size", "pct_of_total", "high_income_rate_pct"]].to_string(index=False))

num_profile = df_profile.groupby("cluster")[
    ["age", "weeks worked in year", "capital gains", "dividends from stocks", "num persons worked for employer"]
].median().round(1)

cat_profile = df_profile.groupby("cluster")[
    ["education", "marital stat", "major occupation code", "sex", "class of worker"]
].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A")

colors = ["#4C9BE8", "#2ECC71", "#E8724C", "#9B59B6", "#F39C12", "#1ABC9C", "#E74C3C", "#95A5A6"]
cluster_colors = [colors[i % len(colors)] for i in range(best_k)]

# PCA scatter — sample 8k for readability
sample_idx = np.random.choice(len(X_pca), size=8000, replace=False)
fig, ax = plt.subplots(figsize=(9, 6))
for k in range(best_k):
    mask = cluster_labels[sample_idx] == k
    ax.scatter(X_pca[sample_idx][mask, 0], X_pca[sample_idx][mask, 1],
               s=8, alpha=0.4, color=colors[k], label=f"Cluster {k}")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("K-Means Clusters — PCA Space (8,000 sample)", fontweight="bold")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "cluster_pca_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# income rate per cluster
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([f"Cluster {i}" for i in size_df["cluster"]], size_df["high_income_rate_pct"],
              color=cluster_colors, edgecolor="white", width=0.6)
ax.axhline(y_full.mean()*100, linestyle="--", color="gray", lw=1.2,
           label=f"Population avg ({y_full.mean()*100:.1f}%)")
ax.set_ylabel(">$50k Rate (%)")
ax.set_title("Income Rate by Cluster", fontweight="bold")
ax.legend()
for bar, val in zip(bars, size_df["high_income_rate_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "cluster_income_rate.png"), dpi=150, bbox_inches="tight")
plt.close()

# age distribution
fig, ax = plt.subplots(figsize=(9, 5))
for k in range(best_k):
    ages = df_profile[df_profile["cluster"] == k]["age"]
    ax.hist(ages, bins=30, alpha=0.5, color=colors[k], label=f"Cluster {k}", density=True)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.set_title("Age Distribution by Cluster", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "cluster_age_dist.png"), dpi=150, bbox_inches="tight")
plt.close()

# education stacked bar
edu_order = [
    "Less than 1st grade", "1st 2nd 3rd or 4th grade", "5th or 6th grade",
    "7th and 8th grade", "9th grade", "10th grade", "11th grade", "12th grade no diploma",
    "High school graduate", "Some college but no degree", "Associates degree-academic program",
    "Associates degree-occup /vocational", "Bachelors degree(BA AB BS)",
    "Masters degree(MA MS MEng MEd MSW MBA)", "Prof school degree (MD DDS DVM LLB JD)",
    "Doctorate degree(PhD EdD)",
]
edu_present = [e for e in edu_order if e in df_profile["education"].values]
edu_df = df_profile.groupby(["cluster", "education"]).size().unstack(fill_value=0)
edu_df = edu_df[[c for c in edu_present if c in edu_df.columns]]
edu_pct = edu_df.div(edu_df.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(best_k)
for i, col in enumerate(edu_pct.columns):
    ax.bar(range(best_k), edu_pct[col], bottom=bottom,
           color=plt.cm.tab20.colors[i % 20], label=col, width=0.6)
    bottom += edu_pct[col].values
ax.set_xticks(range(best_k))
ax.set_xticklabels([f"Cluster {k}" for k in range(best_k)])
ax.set_ylabel("% of Cluster")
ax.set_title("Education Distribution by Cluster", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(SEG_DIR, "cluster_education.png"), dpi=150, bbox_inches="tight")
plt.close()

# save summary
summary = size_df.merge(num_profile.reset_index(), on="cluster").merge(cat_profile.reset_index(), on="cluster")
summary.to_csv(os.path.join(SEG_DIR, "cluster_summary.csv"), index=False)
df_profile[["cluster"]].to_csv(os.path.join(SEG_DIR, "cluster_labels.csv"), index=False)

print(f"\nSaved plots and CSVs to {SEG_DIR}/")
