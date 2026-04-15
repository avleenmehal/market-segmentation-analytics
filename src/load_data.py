import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS_FILE = os.path.join(BASE_DIR, "census-bureau.columns")
DATA_FILE = os.path.join(BASE_DIR, "census-bureau.data")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "raw_data.pkl")
CLEAN_OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "clean_data.pkl")

# 1. Read column names (one name per line, skip blank lines)
with open(COLUMNS_FILE) as f:
    columns = [line.strip() for line in f if line.strip()]

# 2. Load data
df = pd.read_csv(DATA_FILE, header=None, names=columns)

# 3. Strip whitespace from all string columns
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

# 4. Print diagnostics
print("=== Shape ===")
print(df.shape)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== dtypes ===")
print(df.dtypes)

# 5. Save
df.to_pickle(OUTPUT_FILE)
print(f"\nSaved raw DataFrame to {OUTPUT_FILE}")


def clean_labels():
    # 1. Load raw data
    df = pd.read_pickle(OUTPUT_FILE)

    # 2. Rename label column and strip trailing period
    df = df.rename(columns={"label": "income"})
    df["income"] = df["income"].str.rstrip(".")

    # 3. Create binary target: 1 = '50000+', 0 = '- 50000'
    df["target"] = (df["income"] == "50000+").astype(int)

    # 4. Print class distributions
    print("=== Raw income label distribution ===")
    raw_counts = df["income"].value_counts()
    raw_pcts = df["income"].value_counts(normalize=True) * 100
    dist = pd.DataFrame({"count": raw_counts, "percent": raw_pcts.round(2)})
    print(dist.to_string())

    print("\n=== Binary target distribution ===")
    tgt_counts = df["target"].value_counts().sort_index()
    tgt_pcts = df["target"].value_counts(normalize=True).sort_index() * 100
    tgt_dist = pd.DataFrame({
        "label": {0: "- 50000  (0)", 1: "50000+   (1)"},
        "count": tgt_counts,
        "percent": tgt_pcts.round(2),
    })
    print(tgt_dist.to_string())

    # 5. Save
    df.to_pickle(CLEAN_OUTPUT_FILE)
    print(f"\nSaved clean DataFrame to {CLEAN_OUTPUT_FILE}")


if __name__ == "__main__":
    clean_labels()
