# Census Income Classification & Segmentation

Binary classification to predict whether a US Census respondent earns >$50,000/year, plus customer segmentation for retail marketing targeting.

**Data:** US Census Bureau — 199,523 rows, 42 features (1994–95)  
**Problem:** 15:1 class imbalance, mixed numerical and categorical features  
**Best model:** LightGBM — PR-AUC 0.6903, 10.5× lift over random targeting

---

## Project structure

```
├── census-bureau.data        # raw data file (no header)
├── census-bureau.columns     # column names, one per line
├── src/
│   ├── load_data.py          # load raw file, clean labels
│   ├── eda.py                # quick visualisation script
│   ├── preprocess.py         # feature engineering, encoding, train/val/test split
│   ├── train.py              # train 4 models, select best
│   ├── evaluate.py           # final test set evaluation
│   ├── tune_lightgbm.py      # hyperparameter search on best model
│   ├── segment.py            # PCA + K-Means customer segmentation
│   └── profile_segments.py   # deep cluster profiling and visualisations
├── notebooks/
│   └── eda_analysis.ipynb    # full EDA with correlation analysis and findings
├── outputs/
│   ├── EDA_REPORT.md
│   ├── EVALUATION_REPORT.md
│   ├── TUNING_REPORT.md
│   └── SEGMENTATION_REPORT.md
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the pipeline

Run scripts in this order from the project root. Each step saves its outputs to `outputs/` so the next step can pick them up.

### 1. Load and clean the data

```bash
python src/load_data.py
```

Reads `census-bureau.data`, attaches column names, strips whitespace, creates binary target column. Saves `outputs/raw_data.pkl` and `outputs/clean_data.pkl`.

---

### 2. Exploratory data analysis

**Option A — quick plots**
```bash
python src/eda.py
```
Saves distribution and missing value charts to `outputs/eda/`.

**Option B — full analysis (recommended)**
```bash
jupyter notebook notebooks/eda_analysis.ipynb
```
Full EDA with correlation analysis, Cramér V, age deep-dive, financial feature deep-dive, and drop/impute decision table. This is where the findings in `outputs/EDA_REPORT.md` came from.

---

### 3. Preprocessing

```bash
python src/preprocess.py
```

- Drops 6 columns (4 migration cols with 50% missing, `weight`, `year`)
- Imputes `?` as `Unknown` for 4 low-missing columns
- Engineers 5 binary flags (`has_capital_gains`, `has_dividends`, etc.)
- Applies `log1p` + `RobustScaler` to skewed financial columns
- `OrdinalEncoder` for all categorical features
- Stratified 60/20/20 split

Saves `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_*.pkl`, `preprocessor.pkl` to `outputs/preprocessed/`.

---

### 4. Model training

```bash
python src/train.py
```

Trains Logistic Regression, Random Forest, XGBoost, and LightGBM. Tunes classification threshold per model on validation PR curve. Selects best model by PR-AUC.

Saves all model `.pkl` files and `val_results.csv` to `outputs/models/`. Saves PR curves and feature importance plots to `outputs/eda/`.

---

### 5. Hyperparameter tuning (LightGBM)

```bash
python src/tune_lightgbm.py
```

RandomizedSearchCV over 25 combinations, 3-fold CV, scoring on PR-AUC. Compares best found params against Round 1 defaults. Only retrains if improvement exceeds 0.003 PR-AUC.

Saves `lgbm_tuning_results.csv` to `outputs/models/`.

---

### 6. Final evaluation on test set

```bash
python src/evaluate.py
```

Loads `best_model.pkl`, runs once on the held-out test set with threshold fixed from validation. Prints full metrics and confusion matrix.

---

### 7. Customer segmentation

```bash
python src/segment.py
```

Filters to adults (age ≥ 16), runs PCA to 90% variance, sweeps K-Means k=3–7, fits final k=5 model. Saves cluster labels, model, and plots to `outputs/segmentation/`.

---

### 8. Cluster profiling

```bash
python src/profile_segments.py
```

Deep numerical and categorical profiling per cluster. Cross-tabs cluster membership with LightGBM classifier predictions. Saves profile plots to `outputs/segmentation/`.

---

## Outputs

| File | Description |
|---|---|
| `outputs/EDA_REPORT.md` | Data exploration findings, feature decisions, model selection rationale |
| `outputs/EVALUATION_REPORT.md` | Model comparison, metric choices, test set results |
| `outputs/TUNING_REPORT.md` | LightGBM hyperparameter search — Round 1 validation |
| `outputs/SEGMENTATION_REPORT.md` | Cluster profiles, personas, marketing recommendations |
| `outputs/models/best_model.pkl` | Final LightGBM model + threshold |
| `outputs/segmentation/kmeans_model.pkl` | K-Means model + PCA transformer |

---

## Key results

| Model | Val PR-AUC | Val F1 | Threshold |
|---|---|---|---|
| LightGBM | 0.6819 | 0.6168 | 0.857 |
| XGBoost | 0.6789 | 0.6204 | 0.847 |
| Random Forest | 0.6027 | 0.5662 | 0.747 |
| Logistic Regression | 0.5456 | 0.5293 | 0.825 |

**Test set (LightGBM):** PR-AUC 0.6903 · Precision 65.2% · Recall 60.5% · Lift 10.5×

**Segments:** 5 clusters — Established Breadwinners (12.6% income rate) are the primary premium target; Early Career & Students (0.7%) are brand-nurture only.
