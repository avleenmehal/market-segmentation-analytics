import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS_FILE = os.path.join(BASE_DIR, "census-bureau.columns")
DATA_FILE = os.path.join(BASE_DIR, "census-bureau.data")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "raw_data.pkl")
CLEAN_OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "clean_data.pkl")

# read column names, one per line
with open(COLUMNS_FILE) as f:
    columns = [line.strip() for line in f if line.strip()]

df = pd.read_csv(DATA_FILE, header=None, names=columns)

# strip whitespace from string cols — raw file has leading spaces
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

print(f"Loaded: {df.shape}")
df.to_pickle(OUTPUT_FILE)


def clean_labels():
    df = pd.read_pickle(OUTPUT_FILE)

    df = df.rename(columns={"label": "income"})
    df["income"] = df["income"].str.rstrip(".")

    # 1 = '>$50k', 0 = '<=50k'
    df["target"] = (df["income"] == "50000+").astype(int)

    counts = df["target"].value_counts().sort_index()
    pcts = df["target"].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({"count": counts, "pct": pcts.round(2)}).to_string())

    df.to_pickle(CLEAN_OUTPUT_FILE)
    print(f"Saved to {CLEAN_OUTPUT_FILE}")


if __name__ == "__main__":
    clean_labels()
