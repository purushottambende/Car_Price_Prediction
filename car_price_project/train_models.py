# -*- coding: utf-8 -*-
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

"""
train_models.py
===============
Standalone training script that:
  1. Loads and explores the dataset
  2. Builds regression and classification sklearn Pipelines
  3. Trains, evaluates and saves all models
  4. Writes models/metadata.json

Run from the project root:
    python train_models.py
"""

import sys
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── make sure src/ is importable ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.multiclass import OneVsRestClassifier

from src.preprocessing import (
    BrandExtractor, CarAgeFeatureEngineer, NumericCleaner,
    DataFrameImputer, ColumnSelector, build_feature_list,
    build_numeric_features, build_categorical_features,
    iqr_bounds, format_inr,
    COL_CAR_NAME, COL_YEAR, COL_SELLING, COL_PRESENT,
    COL_KMS, COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER,
    COL_BRAND, COL_CAR_AGE, COL_KM_PER_YR, CURRENT_YEAR
)

RANDOM_STATE = 42
MODELS_DIR   = ROOT / "models"
ASSETS_DIR   = ROOT / "assets"
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

DATA_PATH = ROOT / "data" / "car_data_new.csv"
df_raw = pd.read_csv(DATA_PATH)
print(f"Shape: {df_raw.shape}")
print("\nColumn dtypes:\n", df_raw.dtypes)
print("\nMissing values:\n", df_raw.isnull().sum())
print("\nSample:\n", df_raw.head())

# ══════════════════════════════════════════════════════════════════════════════
# 2.  BASIC CLEANING (notebook-side)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASIC CLEANING")
print("=" * 60)

df = df_raw.copy()

# Numeric columns are already numeric in this dataset; coerce anyway for safety
numeric_cols = [COL_YEAR, COL_SELLING, COL_PRESENT, COL_KMS, COL_OWNER]
for c in numeric_cols:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )

# Impute
for c in [COL_YEAR, COL_PRESENT, COL_KMS, COL_OWNER]:
    df[c] = df[c].fillna(df[c].median())
for c in [COL_FUEL, COL_SELLER, COL_TRANS]:
    df[c] = df[c].fillna(df[c].mode()[0])

# Drop rows where target is missing
df = df.dropna(subset=[COL_SELLING])
print(f"Shape after cleaning: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING (notebook-side, for EDA)
# ══════════════════════════════════════════════════════════════════════════════
df[COL_BRAND]     = df[COL_CAR_NAME].str.strip().str.split().str[0].str.lower().str.capitalize()
df[COL_CAR_AGE]   = (CURRENT_YEAR - df[COL_YEAR]).clip(lower=0)
df[COL_KM_PER_YR] = df[COL_KMS] / df[COL_CAR_AGE].replace(0, 1)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  EDA PLOTS (saved to assets/)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA PLOTS")
print("=" * 60)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "coolwarm"

# 4a. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[COL_SELLING], bins=30, color="#4C72B0", edgecolor="white")
axes[0].set_title("Selling Price Distribution")
axes[0].set_xlabel("Price (Lakhs ₹)")
axes[1].hist(np.log1p(df[COL_SELLING]), bins=30, color="#DD8452", edgecolor="white")
axes[1].set_title("Log(Selling Price) Distribution")
axes[1].set_xlabel("log(Price + 1)")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "price_distribution.png", dpi=120)
plt.close()
print("  Saved price_distribution.png")

# 4b. Price vs Year
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df[COL_YEAR], df[COL_SELLING], alpha=0.5, color="#4C72B0")
axes[0].set_title("Price vs Year")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Selling Price (Lakhs)")
axes[1].scatter(df[COL_KMS], df[COL_SELLING], alpha=0.5, color="#DD8452")
axes[1].set_title("Price vs Kms Driven")
axes[1].set_xlabel("Kms Driven"); axes[1].set_ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "price_vs_year_kms.png", dpi=120)
plt.close()
print("  Saved price_vs_year_kms.png")

# 4c. Avg price by brand
brand_price = df.groupby(COL_BRAND)[COL_SELLING].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 5))
brand_price.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Average Selling Price by Brand")
ax.set_ylabel("Avg Price (Lakhs ₹)")
ax.set_xlabel("Brand")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "avg_price_by_brand.png", dpi=120)
plt.close()
print("  Saved avg_price_by_brand.png")

# 4d. Avg price by fuel & transmission
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.groupby(COL_FUEL)[COL_SELLING].mean().sort_values().plot(
    kind="barh", ax=axes[0], color="#55A868")
axes[0].set_title("Avg Price by Fuel Type")
df.groupby(COL_TRANS)[COL_SELLING].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="#C44E52")
axes[1].set_title("Avg Price by Transmission")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "avg_price_fuel_trans.png", dpi=120)
plt.close()
print("  Saved avg_price_fuel_trans.png")

# 4e. Correlation heatmap
num_cols = [COL_YEAR, COL_SELLING, COL_PRESENT, COL_KMS, COL_OWNER, COL_CAR_AGE, COL_KM_PER_YR]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "correlation_heatmap.png", dpi=120)
plt.close()
print("  Saved correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  IQR OUTLIER HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OUTLIER HANDLING")
print("=" * 60)

before = len(df)
for col in [COL_SELLING, COL_KMS, COL_PRESENT]:
    lo, hi = iqr_bounds(df[col])
    df = df[(df[col] >= lo) & (df[col] <= hi)]
print(f"  Rows before: {before}, after IQR filter: {len(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  PRICE CATEGORY (classification target)
# ══════════════════════════════════════════════════════════════════════════════
df["price_category"] = pd.qcut(
    df[COL_SELLING], q=3, labels=["Low", "Medium", "High"]
)
print("\nPrice category distribution:\n", df["price_category"].value_counts())

# Category thresholds for metadata
q33 = float(df[COL_SELLING].quantile(1/3))
q67 = float(df[COL_SELLING].quantile(2/3))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE / TARGET SPLIT
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    COL_CAR_NAME, COL_YEAR, COL_PRESENT, COL_KMS,
    COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER
]

X = df[FEATURE_COLS].copy()
y_reg = df[COL_SELLING].values
y_clf = df["price_category"].values

X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=RANDOM_STATE
)
_, _, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  BUILD PREPROCESSING SUB-PIPELINE (shared)
# ══════════════════════════════════════════════════════════════════════════════
NUM_FEATURES = [COL_YEAR, COL_PRESENT, COL_KMS, COL_OWNER, COL_CAR_AGE, COL_KM_PER_YR]
CAT_FEATURES = [COL_BRAND, COL_FUEL, COL_SELLER, COL_TRANS]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ],
    remainder="drop"
)

def make_base_pipeline(model):
    """Wraps brand extraction, age engineering, and preprocessor with a model."""
    return Pipeline([
        ("brand",     BrandExtractor()),
        ("age",       CarAgeFeatureEngineer()),
        ("preprocess", preprocessor),
        ("model",     model),
    ])

# ══════════════════════════════════════════════════════════════════════════════
# 9.  REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REGRESSION MODELS")
print("=" * 60)

def eval_regressor(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    mse  = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, y_pred)
    cv   = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean()
    print(f"  {name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  CV-R2={cv:.3f}")
    return {"model": name, "MAE": round(mae,4), "MSE": round(mse,4),
            "RMSE": round(rmse,4), "R2": round(r2,4), "CV_R2": round(cv,4)}, pipe

reg_results = []

# -- Simple Linear Regression
r, p = eval_regressor("Simple Linear (Year)", make_base_pipeline(LinearRegression()),
                       X_train[[COL_CAR_NAME, COL_YEAR, COL_PRESENT, COL_KMS,
                                  COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER]],
                       yr_train,
                       X_test[[COL_CAR_NAME, COL_YEAR, COL_PRESENT, COL_KMS,
                                COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER]],
                       yr_test)
reg_results.append(r)

# -- Multiple Linear Regression
r, pipe_mlr = eval_regressor("Multiple Linear",
                              make_base_pipeline(LinearRegression()),
                              X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# -- Polynomial Regression (degree 2 on numeric only)
poly_pipe = Pipeline([
    ("brand",      BrandExtractor()),
    ("age",        CarAgeFeatureEngineer()),
    ("preprocess", preprocessor),
    ("poly",       PolynomialFeatures(degree=2, include_bias=False)),
    ("model",      LinearRegression()),
])
r, pipe_poly = eval_regressor("Polynomial (deg=2)", poly_pipe, X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# -- Ridge
alphas = [0.01, 0.1, 1, 10, 100]
best_ridge = None; best_ridge_score = -999
for a in alphas:
    p_ = make_base_pipeline(Ridge(alpha=a, random_state=RANDOM_STATE))
    s  = cross_val_score(p_, X_train, yr_train, cv=5, scoring="r2").mean()
    if s > best_ridge_score:
        best_ridge_score = s; best_ridge = a
r, pipe_ridge = eval_regressor(f"Ridge (alpha={best_ridge})",
                                make_base_pipeline(Ridge(alpha=best_ridge, random_state=RANDOM_STATE)),
                                X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# -- Lasso
best_lasso = None; best_lasso_score = -999
for a in alphas:
    p_ = make_base_pipeline(Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=5000))
    s  = cross_val_score(p_, X_train, yr_train, cv=5, scoring="r2").mean()
    if s > best_lasso_score:
        best_lasso_score = s; best_lasso = a
r, pipe_lasso = eval_regressor(f"Lasso (alpha={best_lasso})",
                                make_base_pipeline(Lasso(alpha=best_lasso, random_state=RANDOM_STATE, max_iter=5000)),
                                X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# -- ElasticNet
r, pipe_en = eval_regressor("ElasticNet",
                             make_base_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5,
                                                            random_state=RANDOM_STATE, max_iter=5000)),
                             X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# -- Random Forest Regression (≤100 trees)
r, pipe_rfr = eval_regressor("Random Forest Reg",
                              make_base_pipeline(RandomForestRegressor(
                                  n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
                              X_train, yr_train, X_test, yr_test)
reg_results.append(r)

# Pick best regressor by R2
reg_df = pd.DataFrame(reg_results)
best_reg_row = reg_df.loc[reg_df["R2"].idxmax()]
print(f"\nBest regressor: {best_reg_row['model']}  R2={best_reg_row['R2']}")

# Build final best regressor pipeline and fit on full train set
pipe_best_reg = make_base_pipeline(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
)
pipe_best_reg.fit(X_train, yr_train)

# Actual vs Predicted plot
y_pred_best = pipe_best_reg.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(yr_test, y_pred_best, alpha=0.6, color="#4C72B0")
lims = [min(yr_test.min(), y_pred_best.min()), max(yr_test.max(), y_pred_best.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("Actual Price"); axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted (Best Regressor)")
residuals = yr_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.6, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted Price"); axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "regression_actual_vs_pred.png", dpi=120)
plt.close()
print("  Saved regression_actual_vs_pred.png")

# Regression comparison bar chart
fig, ax = plt.subplots(figsize=(12, 5))
reg_df_plot = reg_df.sort_values("R2", ascending=False)
bars = ax.barh(reg_df_plot["model"], reg_df_plot["R2"], color="#4C72B0", edgecolor="white")
ax.set_xlabel("R² Score")
ax.set_title("Regression Model Comparison (R² Score)")
for bar, val in zip(bars, reg_df_plot["R2"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "regression_comparison.png", dpi=120)
plt.close()
print("  Saved regression_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10.  CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CLASSIFICATION MODELS")
print("=" * 60)

LABELS = ["Low", "Medium", "High"]

clf_results = []

def eval_classifier(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    rpt = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    macro_p  = rpt["macro avg"]["precision"]
    macro_r  = rpt["macro avg"]["recall"]
    macro_f1 = rpt["macro avg"]["f1-score"]
    # ROC AUC one-vs-rest
    try:
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_te)
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_te, classes=pipe.classes_)
            auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        else:
            y_prob = None
            auc = float("nan")
    except Exception:
        auc = float("nan")

    print(f"  {name:28s}  Acc={acc:.3f}  F1={macro_f1:.3f}  AUC={auc:.3f}")
    return {
        "model": name,
        "Accuracy": round(acc, 4),
        "Macro_Precision": round(macro_p, 4),
        "Macro_Recall": round(macro_r, 4),
        "Macro_F1": round(macro_f1, 4),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else None,
    }, pipe


classifiers = [
    ("Logistic Regression",    make_base_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))),
    ("KNN (k=5)",              make_base_pipeline(KNeighborsClassifier(n_neighbors=5))),
    ("Decision Tree",          make_base_pipeline(DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE))),
    ("Random Forest Clf",      make_base_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1))),
    ("SVM (RBF)",              make_base_pipeline(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))),
    ("Gaussian NB",            make_base_pipeline(GaussianNB())),
]

best_clf_pipe = None; best_clf_f1 = -1; best_clf_name = ""
for name, pipe in classifiers:
    r, trained_pipe = eval_classifier(name, pipe, X_train, yc_train, X_test, yc_test)
    clf_results.append(r)
    if r["Macro_F1"] > best_clf_f1:
        best_clf_f1 = r["Macro_F1"]; best_clf_pipe = trained_pipe; best_clf_name = name

clf_df = pd.DataFrame(clf_results)
print(f"\nBest classifier: {best_clf_name}  F1={best_clf_f1:.3f}")

# Confusion matrix for best classifier
y_pred_clf = best_clf_pipe.predict(X_test)
cm = confusion_matrix(yc_test, y_pred_clf, labels=LABELS)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
ax.set_title(f"Confusion Matrix – {best_clf_name}")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print("  Saved confusion_matrix.png")

# ROC curves (one-vs-rest)
fig, ax = plt.subplots(figsize=(8, 6))
color_map = {"Low": "#4C72B0", "Medium": "#DD8452", "High": "#55A868"}
clf_classes = list(best_clf_pipe.classes_)
for label in LABELS:
    try:
        if hasattr(best_clf_pipe, "predict_proba") and label in clf_classes:
            col_idx = clf_classes.index(label)
            y_prob_col = best_clf_pipe.predict_proba(X_test)[:, col_idx]
            y_true_binary = (np.array(yc_test) == label).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_col)
            auc_val = roc_auc_score(y_true_binary, y_prob_col)
            ax.plot(fpr, tpr, color=color_map.get(label, "#333333"), label=f"{label} (AUC={auc_val:.2f})")
    except Exception:
        pass
ax.plot([0,1],[0,1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves – {best_clf_name}")
ax.legend()
plt.tight_layout()
plt.savefig(ASSETS_DIR / "roc_curves.png", dpi=120)
plt.close()
print("  Saved roc_curves.png")

# Classification comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
clf_df_plot = clf_df.sort_values("Macro_F1", ascending=False)
bars = ax.barh(clf_df_plot["model"], clf_df_plot["Macro_F1"], color="#55A868", edgecolor="white")
ax.set_xlabel("Macro F1 Score")
ax.set_title("Classification Model Comparison (Macro F1 Score)")
for bar, val in zip(bars, clf_df_plot["Macro_F1"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "classification_comparison.png", dpi=120)
plt.close()
print("  Saved classification_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11.  SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

joblib.dump(pipe_best_reg, MODELS_DIR / "regressor.joblib", compress=3)
joblib.dump(best_clf_pipe, MODELS_DIR / "classifier.joblib", compress=3)

# File sizes
for fname in ["regressor.joblib", "classifier.joblib"]:
    sz = (MODELS_DIR / fname).stat().st_size / 1024
    print(f"  {fname}: {sz:.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# 12.  BUILD METADATA.JSON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("WRITING METADATA")
print("=" * 60)

brands_list = sorted(df[COL_BRAND].dropna().unique().tolist())
fuels_list  = sorted(df[COL_FUEL].dropna().unique().tolist())
sellers_list = sorted(df[COL_SELLER].dropna().unique().tolist())
trans_list  = sorted(df[COL_TRANS].dropna().unique().tolist())
owners_list = sorted(df[COL_OWNER].dropna().unique().tolist())

metadata = {
    "sklearn_version": sklearn.__version__,
    "current_year": CURRENT_YEAR,
    "feature_list": FEATURE_COLS,
    "categorical_options": {
        COL_BRAND:  brands_list,
        COL_FUEL:   fuels_list,
        COL_SELLER: sellers_list,
        COL_TRANS:  trans_list,
        COL_OWNER:  [int(o) for o in owners_list],
    },
    "numeric_ranges": {
        COL_YEAR:    {"min": int(df[COL_YEAR].min()),    "max": int(df[COL_YEAR].max()),    "default": int(df[COL_YEAR].median())},
        COL_PRESENT: {"min": float(df[COL_PRESENT].min()), "max": float(df[COL_PRESENT].max()), "default": float(df[COL_PRESENT].median())},
        COL_KMS:     {"min": float(df[COL_KMS].min()),   "max": float(df[COL_KMS].max()),   "default": float(df[COL_KMS].median())},
        COL_OWNER:   {"min": int(df[COL_OWNER].min()),   "max": int(df[COL_OWNER].max()),   "default": int(df[COL_OWNER].median())},
    },
    "category_thresholds": {
        "Low_to_Medium_lakhs": round(q33, 2),
        "Medium_to_High_lakhs": round(q67, 2),
    },
    "regression_results": reg_df.to_dict(orient="records"),
    "classification_results": clf_df.to_dict(orient="records"),
    "best_regressor": best_reg_row["model"],
    "best_classifier": best_clf_name,
}

with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("  Saved models/metadata.json")
print("\n" + "=" * 60)
print("TRAINING COMPLETE ✓")
print("=" * 60)
