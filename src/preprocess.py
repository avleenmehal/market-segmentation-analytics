import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA = os.path.join(BASE_DIR, "outputs", "clean_data.pkl")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_pickle(CLEAN_DATA)
print(f"Loaded: {df.shape}")

# drop columns decided in EDA — migration cols (50% missing, near-zero signal),
# weight (sampling weight, not a person feature), year (only 2 values)
DROP_COLS = [
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "migration prev res in sunbelt",
    "weight",
    "year",
]
df = df.drop(columns=DROP_COLS)

# impute '?' as 'Unknown' — these columns have small % missing, random pattern
IMPUTE_COLS = [
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "state of previous residence",
]
for col in IMPUTE_COLS:
    df[col] = df[col].replace("?", "Unknown")

# catch pandas NaN values that slipped through (e.g. hispanic origin had 874 NaN)
cat_cols_all = df.select_dtypes(include="object").columns.tolist()
num_cols_all = df.select_dtypes(include="number").columns.tolist()

for col in cat_cols_all:
    if df[col].isna().any():
        df[col] = df[col].fillna("Unknown")

for col in num_cols_all:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# binary flags — presence of investment income is stronger signal than raw amount
# (columns are 89-98% zeros, the threshold effect matters more than the value)
df["has_capital_gains"]  = (df["capital gains"] > 0).astype(int)
df["has_capital_losses"] = (df["capital losses"] > 0).astype(int)
df["has_dividends"]      = (df["dividends from stocks"] > 0).astype(int)
df["has_wage"]           = (df["wage per hour"] > 0).astype(int)
df["is_fulltime_year"]   = (df["weeks worked in year"] == 52).astype(int)

X = df.drop(columns=["income", "target"])
y = df["target"]

# log1p before scaling — these are heavily right-skewed (skewness 9-28)
LOG_NUM_COLS = ["capital gains", "capital losses", "dividends from stocks", "wage per hour"]

NUM_COLS = [
    "age", "weeks worked in year", "num persons worked for employer",
    "own business or self employed", "veterans benefits",
    "detailed industry recode", "detailed occupation recode",
    "has_capital_gains", "has_capital_losses", "has_dividends",
    "has_wage", "is_fulltime_year",
]

# everything remaining as object dtype
CAT_COLS = X.select_dtypes(include="object").columns.tolist()

# stratified split — preserves 6.2% positive rate in all three sets
# step 1: carve out 20% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
# step 2: 75/25 on the rest gives 60/20/20 overall
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

for name, X_s, y_s in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    pos = y_s.sum()
    print(f"  {name}: {len(X_s):,} rows | pos={pos:,} ({pos/len(y_s)*100:.2f}%)")

log_num_pipeline = Pipeline([
    ("log1p",  FunctionTransformer(np.log1p, validate=False)),
    ("scaler", RobustScaler()),
])

num_pipeline  = Pipeline([("scaler", RobustScaler())])

# unknown_value=-1 handles categories unseen at fit time
cat_pipeline  = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("log_num", log_num_pipeline, LOG_NUM_COLS),
        ("num",     num_pipeline,     NUM_COLS),
        ("cat",     cat_pipeline,     CAT_COLS),
    ],
    remainder="drop",
)

# fit on train only — val and test are transformed with the same fitted scaler/encoder
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

print(f"Feature count after preprocessing: {X_train_proc.shape[1]}")

feature_names = LOG_NUM_COLS + NUM_COLS + CAT_COLS

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_proc)
np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val_proc)
np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test_proc)

y_train.to_pickle(os.path.join(OUT_DIR, "y_train.pkl"))
y_val.to_pickle(os.path.join(OUT_DIR, "y_val.pkl"))
y_test.to_pickle(os.path.join(OUT_DIR, "y_test.pkl"))

with open(os.path.join(OUT_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)

with open(os.path.join(OUT_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

print(f"Saved to {OUT_DIR}/")
